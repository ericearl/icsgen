"""icsgen command-line interface.

Subcommands:
    icsgen login    — prompt for provider/endpoint/key/model, store config
    icsgen x        — generate an ICS file from one or more prompts
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import datetime
from pathlib import Path

from icsgen import __version__
from icsgen.config import (
    PROVIDER_DEFAULTS,
    PROVIDERS,
    Config,
    ConfigError,
    ProviderConfig,
    config_path,
    load_config,
    save_config,
    warn_if_world_readable,
)
from icsgen.ics_builder import (
    IcsBuildError,
    build_calendar,
    parse_events,
    parse_llm_response,
    write_calendar,
)
from icsgen.prompts import build_system_prompt, build_user_message
from icsgen.providers import get_provider_client
from icsgen.providers.base import ProviderError


# ---------------------------------------------------------------------------
# argparse setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icsgen",
        description=(
            "Generate ICS calendar files from natural-language prompts via "
            "Claude, ChatGPT, or Gemini."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # --- login -----------------------------------------------------------
    p_login = sub.add_parser(
        "login",
        help="Configure an LLM provider (endpoint, API key, model).",
    )
    p_login.set_defaults(func=cmd_login)

    # --- x ---------------------------------------------------------------
    p_x = sub.add_parser(
        "x",
        help="Generate an ICS file from prompts.",
        description=(
            "Generate an ICS calendar file. The required positional argument is "
            "the 'center time' — the anchor event. Additional events can be "
            "passed via -a/--add/--additional. Each quoted string becomes one "
            "appointment in the output."
        ),
    )
    p_x.add_argument(
        "center_time",
        metavar="CENTER_TIME",
        help="Quotation-wrapped natural-language description of the anchor event.",
    )
    p_x.add_argument(
        "-a", "--add", "--additional",
        dest="additional",
        nargs="+",
        default=[],
        metavar="PROMPT",
        help="One or more additional event prompts, each quoted.",
    )
    p_x.add_argument(
        "-o", "--output",
        default="icsgen-output.ics",
        metavar="PATH",
        help="Output .ics file path. Default: icsgen-output.ics",
    )
    p_x.add_argument(
        "-p", "--provider",
        choices=PROVIDERS,
        default=None,
        help="Override the active provider for this call.",
    )
    p_x.add_argument(
        "-m", "--model",
        default=None,
        metavar="MODEL",
        help="Override the model string for the chosen provider.",
    )
    p_x.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parsed event JSON to stdout; do not write a file.",
    )
    p_x.add_argument(
        "--save-json",
        default=None,
        metavar="PATH",
        help="Also write the raw LLM-returned JSON to this path.",
    )
    p_x.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print the system prompt, user message, and raw LLM response.",
    )
    p_x.set_defaults(func=cmd_x)

    return parser


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str | None = None, *, secret: bool = False) -> str:
    """Prompt the user, with optional default and optional secret-masking."""
    if default:
        suffix = f" [{default}]"
    else:
        suffix = ""
    prompt_text = f"{label}{suffix}: "

    if secret:
        value = getpass.getpass(prompt_text)
    else:
        value = input(prompt_text)
    value = value.strip()
    if not value and default is not None:
        return default
    return value


def _detect_timezone() -> str:
    """Best-effort guess of the user's IANA timezone."""
    try:
        # Python 3.9+: datetime.now().astimezone() returns a tzinfo with .key on POSIX.
        tz = datetime.now().astimezone().tzinfo
        name = getattr(tz, "key", None) or str(tz)
        if name in {"UTC", "Coordinated Universal Time"} or not name:
            return "UTC"
        return name
    except Exception:
        return "UTC"


# ---------------------------------------------------------------------------
# subcommand: login
# ---------------------------------------------------------------------------

def cmd_login(args: argparse.Namespace) -> int:
    """Interactive prompt to configure a provider."""
    print("icsgen login — configure an LLM provider.")
    print(f"Config file: {config_path()}\n")

    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Provider
    valid = ", ".join(PROVIDERS)
    while True:
        provider = _prompt(f"Provider [{valid}]").lower()
        if provider in PROVIDERS:
            break
        print(f"  Please choose one of: {valid}")

    defaults = PROVIDER_DEFAULTS[provider]  # type: ignore[index]

    # If we're re-configuring an existing provider, prefer its current values
    # as defaults so re-running login is idempotent for simple edits.
    existing = cfg.providers.get(provider)  # type: ignore[arg-type]
    endpoint_default = existing.endpoint if existing else defaults["endpoint"]
    model_default = existing.model if existing else defaults["model"]

    endpoint = _prompt("API endpoint", default=endpoint_default)
    api_key = _prompt("API key", secret=True)
    if not api_key:
        print("error: API key cannot be empty.", file=sys.stderr)
        return 2
    model = _prompt("Model", default=model_default)

    cfg.providers[provider] = ProviderConfig(  # type: ignore[index]
        name=provider,  # type: ignore[arg-type]
        endpoint=endpoint,
        api_key=api_key,
        model=model,
    )

    # Active?
    is_only = len(cfg.providers) == 1
    make_active_default = "Y" if is_only or cfg.active_provider == provider else "n"
    answer = _prompt(f"Set as active provider? [Y/n]", default=make_active_default).lower()
    if answer in {"y", "yes", ""}:
        cfg.active_provider = provider  # type: ignore[assignment]
    elif cfg.active_provider is None:
        # No active provider was set and the user said no — still make this one active.
        cfg.active_provider = provider  # type: ignore[assignment]

    path = save_config(cfg)
    print(f"\nSaved to {path}")
    print(f"Active provider: {cfg.active_provider}")
    return 0


# ---------------------------------------------------------------------------
# subcommand: x
# ---------------------------------------------------------------------------

def cmd_x(args: argparse.Namespace) -> int:
    """Generate an ICS file from one center-time prompt + optional additional prompts."""
    warn_if_world_readable()

    try:
        cfg = load_config()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Choose provider config
    try:
        if args.provider:
            pcfg = cfg.get(args.provider)
        else:
            pcfg = cfg.get_active()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Per-call model override
    if args.model:
        pcfg = ProviderConfig(
            name=pcfg.name,
            endpoint=pcfg.endpoint,
            api_key=pcfg.api_key,
            model=args.model,
        )

    # Build prompts
    timezone = _detect_timezone()
    system_prompt = build_system_prompt(timezone)
    user_message = build_user_message(args.center_time, args.additional, timezone)

    if args.verbose:
        print("--- system prompt ---", file=sys.stderr)
        print(system_prompt, file=sys.stderr)
        print("--- user message ---", file=sys.stderr)
        print(user_message, file=sys.stderr)
        print(f"--- provider: {pcfg.name} model: {pcfg.model} ---", file=sys.stderr)

    # Call LLM
    client = get_provider_client(pcfg)
    try:
        raw = client.complete(system_prompt, user_message)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    if args.verbose:
        print("--- raw response ---", file=sys.stderr)
        print(raw, file=sys.stderr)

    if args.save_json:
        Path(args.save_json).write_text(raw, encoding="utf-8")
        if args.verbose:
            print(f"Wrote raw JSON to {args.save_json}", file=sys.stderr)

    # Parse + build
    try:
        payload = parse_llm_response(raw)
        events = parse_events(payload)
    except IcsBuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4

    if args.dry_run:
        # Re-emit the validated event payload in canonical form so the user sees
        # exactly what would have been written.
        out = {
            "events": [
                {
                    "summary": ev.summary,
                    "all_day": ev.all_day,
                    "start": ev.start.isoformat(),
                    "end": ev.end.isoformat(),
                    "location": ev.location,
                    "description": ev.description,
                    "timezone": ev.timezone,
                }
                for ev in events
            ]
        }
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    cal = build_calendar(events)
    try:
        write_calendar(cal, args.output)
    except OSError as e:
        print(f"error: failed to write {args.output}: {e}", file=sys.stderr)
        return 5

    print(f"Wrote {len(events)} event(s) to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
