"""Smoke tests for icsgen — no network, no LLM calls."""

from __future__ import annotations

import json

import pytest

from icsgen.ics_builder import (
    IcsBuildError,
    build_calendar,
    parse_events,
    parse_llm_response,
)
from icsgen.prompts import build_system_prompt, build_user_message


# --- parse_llm_response ------------------------------------------------------

def test_parses_plain_json():
    text = '{"events": []}'
    assert parse_llm_response(text) == {"events": []}


def test_parses_json_inside_fences():
    text = "```json\n{\"events\": []}\n```"
    assert parse_llm_response(text) == {"events": []}


def test_parses_json_with_surrounding_prose():
    text = "Sure! Here you go:\n{\"events\": [{\"x\": 1}]}\nLet me know!"
    assert parse_llm_response(text) == {"events": [{"x": 1}]}


def test_raises_on_no_json():
    with pytest.raises(IcsBuildError):
        parse_llm_response("no json here, sorry")


# --- parse_events + build_calendar ------------------------------------------

def _sample_payload() -> dict:
    return {
        "events": [
            {
                "summary": "Presentation",
                "all_day": False,
                "start": "2026-06-02T12:00:00",
                "end": "2026-06-02T13:00:00",
                "location": "Conference Room A",
                "description": None,
                "timezone": "America/New_York",
            },
            {
                "summary": "Prep for presentation",
                "all_day": True,
                "start": "2026-05-19",
                "end": "2026-05-20",
                "location": None,
                "description": None,
                "timezone": "America/New_York",
            },
        ]
    }


def test_parse_events_returns_one_per_input():
    events = parse_events(_sample_payload())
    assert len(events) == 2
    assert events[0].summary == "Presentation"
    assert events[0].location == "Conference Room A"
    assert events[1].all_day is True


def test_build_calendar_emits_valid_ics(tmp_path):
    events = parse_events(_sample_payload())
    cal = build_calendar(events)
    text = "".join(cal.serialize_iter())
    assert "BEGIN:VCALENDAR" in text
    assert "END:VCALENDAR" in text
    assert text.count("BEGIN:VEVENT") == 2
    assert "Presentation" in text
    assert "Prep for presentation" in text


def test_missing_required_field_raises():
    payload = {"events": [{"summary": "x"}]}
    with pytest.raises(IcsBuildError):
        parse_events(payload)


def test_invalid_timezone_raises():
    payload = {
        "events": [
            {
                "summary": "x",
                "all_day": False,
                "start": "2026-06-02T12:00:00",
                "end": "2026-06-02T13:00:00",
                "location": None,
                "description": None,
                "timezone": "Not/A_Real_Zone",
            }
        ]
    }
    with pytest.raises(IcsBuildError):
        parse_events(payload)


# --- prompt construction -----------------------------------------------------

def test_system_prompt_includes_timezone_and_today():
    s = build_system_prompt("America/New_York")
    assert "America/New_York" in s
    # The prompt mentions the JSON schema and the strict rules.
    assert "events" in s
    assert "all_day" in s
    assert "summary" in s


def test_user_message_numbers_prompts_and_marks_anchor():
    msg = build_user_message("anchor here", ["second", "third"])
    assert "1. ANCHOR" in msg
    assert "anchor here" in msg
    assert "2. ADDITIONAL: second" in msg
    assert "3. ADDITIONAL: third" in msg


def test_user_message_handles_only_anchor():
    msg = build_user_message("just one", [])
    assert "1. ANCHOR" in msg
    assert "2." not in msg
