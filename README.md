# icsgen

Generate `.ics` calendar files from natural-language prompts, powered by your choice of LLM (Claude, ChatGPT, or Gemini).

You hand it a description of an appointment — anything from `"Presentation on Tuesday June 2nd at 12pm for an hour in Conference Room A"` to `"reminder all day 2 weeks before the center time"` — and `icsgen` asks an LLM to resolve it into structured event data, then writes a valid `.ics` file you can import into any calendar app.

## Install

```bash
# with uv (recommended)
uv tool install icsgen

# or with pip
pip install icsgen
```

## Configure a provider

Run `icsgen login` once per provider you want to use. It prompts for the provider name, API endpoint, API key (masked), and default model, then asks whether to set it as the active provider.

```bash
$ icsgen login
Provider [claude / openai / gemini]: claude
API endpoint [https://api.anthropic.com/v1/messages]:
API key: ********
Model [claude-sonnet-4-6]:
Set as active provider? [Y/n]: y
Saved. Active provider: claude
```

All three providers can be configured simultaneously; switch which is active by re-running `icsgen login` for a different provider and answering `y`, or override per-invocation with `--provider`.

Config lives at `~/.config/icsgen/config.toml` with `0600` permissions.

## Generate a calendar

The required positional argument is the **center time** — the anchor event. Any number of additional events can be passed via `-a / --add / --additional`, each as a separate quoted string. The center event is parsed first, and additional events may reference it ("2 weeks before the center time", "the day after").

```bash
icsgen x "Presentation for Tuesday June 2nd at 12pm for an hour" \
  --add "all-day reminder titled 'Prep for presentation' 2 weeks before the center time" \
       "30-min dry run the day before at 4pm in my office"
```

Outputs `icsgen-output.ics` in the current directory by default. Override with `-o`:

```bash
icsgen x "Lunch with Sarah next Friday" -o lunch.ics
```

### Flags

| Flag | Description |
| --- | --- |
| `-a / --add / --additional` | One or more additional event prompts (each quoted). |
| `-o / --output PATH` | Output file path. Default: `icsgen-output.ics`. |
| `-p / --provider {claude,openai,gemini}` | Override the active provider for this call. |
| `-m / --model MODEL` | Override the model string for the chosen provider. |
| `--dry-run` | Print the parsed event JSON to stdout; do not write a file. |
| `--save-json PATH` | Also write the raw LLM-returned JSON to this path. |
| `-v / --verbose` | Print the system prompt, user message, and raw LLM response. |

## How it works

1. `icsgen` loads your active provider config.
2. It builds a system prompt that tells the LLM how to interpret the inputs and asks for strict JSON in a fixed schema.
3. It sends the center-time prompt + each additional prompt as a single user message.
4. The LLM returns one event per input prompt as JSON.
5. `icsgen` validates the JSON and writes a single `.ics` containing all events.

## Development

```bash
git clone https://github.com/ericearl/icsgen
cd icsgen
uv sync --all-extras
uv run pytest
```

To build a distribution:

```bash
uv build
```

To publish to PyPI (requires PyPI credentials):

```bash
uv publish
```
