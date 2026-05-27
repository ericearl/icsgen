"""Parse the LLM's JSON response and build a single ICS calendar.

We accept slightly forgiving input (JSON sometimes wrapped in markdown fences,
trailing prose) and convert it into events using the `ics` library. Validation
errors raise IcsBuildError so the CLI can show a friendly message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from dateutil import parser as dateparser
from ics import Calendar, Event
from zoneinfo import ZoneInfo


class IcsBuildError(Exception):
    """Raised when the LLM output cannot be turned into a valid ICS file."""


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model added them despite instructions."""
    return _FENCE_RE.sub("", text).strip()


def _extract_json_object(text: str) -> str:
    """Find the outermost {...} JSON object in `text`.

    Used as a last-resort fallback when the model wraps its JSON in prose.
    """
    start = text.find("{")
    if start == -1:
        raise IcsBuildError(f"No JSON object found in response:\n{text}")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise IcsBuildError(f"Unbalanced JSON object in response:\n{text}")


def parse_llm_response(text: str) -> dict[str, Any]:
    """Parse the LLM's text response into a Python dict.

    Tries: (1) direct json.loads, (2) strip ``` fences then json.loads,
    (3) extract the first balanced {...} then json.loads.
    """
    candidates = [text, _strip_fences(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    # Last resort
    extracted = _extract_json_object(text)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as e:
        raise IcsBuildError(f"Could not parse JSON from LLM response: {e}\n{text}") from e


@dataclass
class ParsedEvent:
    """An event as understood from the LLM, ready to be added to a Calendar."""

    summary: str
    all_day: bool
    start: datetime | date
    end: datetime | date
    location: str | None
    description: str | None
    timezone: str


def _parse_datetime(value: str, tz: ZoneInfo, *, all_day: bool) -> datetime | date:
    """Parse an ISO date or datetime from the LLM.

    For all-day events we return a `date`. For timed events we return a
    timezone-aware `datetime` in the given zone.
    """
    if not isinstance(value, str):
        raise IcsBuildError(f"Expected ISO string, got {type(value).__name__}: {value!r}")
    try:
        dt = dateparser.isoparse(value)
    except (ValueError, TypeError) as e:
        raise IcsBuildError(f"Could not parse datetime {value!r}: {e}") from e
    if all_day:
        # If the model gave a full datetime for an all-day event, drop the time.
        return dt.date() if isinstance(dt, datetime) else dt
    # Timed event: attach timezone if the model omitted it (as instructed).
    if isinstance(dt, date) and not isinstance(dt, datetime):
        # Should not happen for a timed event, but coerce safely.
        raise IcsBuildError(
            f"Got date-only value {value!r} for a timed event; expected datetime."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def _normalize_event(raw: dict[str, Any]) -> ParsedEvent:
    """Validate one event dict from the LLM and return a ParsedEvent."""
    if not isinstance(raw, dict):
        raise IcsBuildError(f"Each event must be an object, got {type(raw).__name__}")

    try:
        summary = raw["summary"]
        all_day = bool(raw["all_day"])
        start_raw = raw["start"]
        end_raw = raw["end"]
        tz_name = raw["timezone"]
    except KeyError as e:
        raise IcsBuildError(f"Event missing required field {e}: {raw!r}") from e

    if not isinstance(summary, str) or not summary.strip():
        raise IcsBuildError(f"Event 'summary' must be a non-empty string: {raw!r}")

    try:
        tz = ZoneInfo(tz_name)
    except Exception as e:
        raise IcsBuildError(f"Invalid IANA timezone {tz_name!r}: {e}") from e

    start = _parse_datetime(start_raw, tz, all_day=all_day)
    end = _parse_datetime(end_raw, tz, all_day=all_day)

    location = raw.get("location")
    if location is not None and not isinstance(location, str):
        location = str(location)
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        description = str(description)

    return ParsedEvent(
        summary=summary.strip(),
        all_day=all_day,
        start=start,
        end=end,
        location=location,
        description=description,
        timezone=tz_name,
    )


def parse_events(payload: dict[str, Any]) -> list[ParsedEvent]:
    """Validate the top-level payload and return a list of ParsedEvents."""
    if "events" not in payload or not isinstance(payload["events"], list):
        raise IcsBuildError(f"Payload must have an 'events' array: {payload!r}")
    if not payload["events"]:
        raise IcsBuildError("LLM returned zero events.")
    return [_normalize_event(e) for e in payload["events"]]


def build_calendar(events: list[ParsedEvent]) -> Calendar:
    """Build an `ics.Calendar` from parsed events."""
    cal = Calendar()
    cal.creator = "-//icsgen//EN"

    for ev in events:
        ical_event = Event()
        ical_event.name = ev.summary
        if ev.location:
            ical_event.location = ev.location
        if ev.description:
            ical_event.description = ev.description

        if ev.all_day:
            # ics's all-day handling: assign a date to begin then call make_all_day().
            ical_event.begin = ev.start.isoformat() if isinstance(ev.start, date) else ev.start
            ical_event.make_all_day()
            # If the model gave an explicit end > start+1 day, we still want a
            # multi-day all-day span. ics handles end via .end for non-all-day,
            # but for all-day we approximate by adding the day delta.
            if isinstance(ev.end, date) and isinstance(ev.start, date):
                # ics's make_all_day defaults end = start + 1 day; if model says
                # something different, we honor it.
                delta = (ev.end - ev.start).days
                if delta > 1:
                    ical_event.end = (ev.start + timedelta(days=delta)).isoformat()
        else:
            ical_event.begin = ev.start
            ical_event.end = ev.end

        cal.events.add(ical_event)

    return cal


def write_calendar(cal: Calendar, output_path: str) -> None:
    """Serialize and write the calendar to disk."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.writelines(cal.serialize_iter())
