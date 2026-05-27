"""Credential and configuration storage for icsgen.

Layout of `~/.config/icsgen/config.toml`:

    active_provider = "claude"

    [providers.claude]
    endpoint = "https://api.anthropic.com/v1/messages"
    api_key = "sk-ant-..."
    model = "claude-sonnet-4-6"

    [providers.openai]
    endpoint = "https://api.openai.com/v1/chat/completions"
    api_key = "sk-..."
    model = "gpt-4o"

    [providers.gemini]
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
    api_key = "..."
    model = "gemini-2.0-flash"

The file is created with mode 0600. All reads and writes go through the
functions in this module.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tomli_w


Provider = Literal["claude", "openai", "gemini"]
PROVIDERS: tuple[Provider, ...] = ("claude", "openai", "gemini")


PROVIDER_DEFAULTS: dict[Provider, dict[str, str]] = {
    "claude": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-6",
    },
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
    },
    "gemini": {
        # Gemini's endpoint includes the model in the URL; we build it dynamically.
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "model": "gemini-2.0-flash",
    },
}


@dataclass
class ProviderConfig:
    """Stored config for a single provider."""

    name: Provider
    endpoint: str
    api_key: str
    model: str


@dataclass
class Config:
    """Full icsgen config (all providers + which is active)."""

    active_provider: Provider | None
    providers: dict[Provider, ProviderConfig]

    def get_active(self) -> ProviderConfig:
        if self.active_provider is None or self.active_provider not in self.providers:
            raise ConfigError(
                "No active provider configured. Run `icsgen login` to set one up."
            )
        return self.providers[self.active_provider]

    def get(self, name: Provider) -> ProviderConfig:
        if name not in self.providers:
            raise ConfigError(
                f"Provider '{name}' is not configured. Run `icsgen login` for it first."
            )
        return self.providers[name]


class ConfigError(Exception):
    """Raised on missing or malformed config."""


def config_dir() -> Path:
    """Return the icsgen config directory, honoring XDG_CONFIG_HOME."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "icsgen"


def config_path() -> Path:
    return config_dir() / "config.toml"


def load_config() -> Config:
    """Load config from disk. Returns an empty Config if the file does not exist."""
    path = config_path()
    if not path.exists():
        return Config(active_provider=None, providers={})

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ConfigError(f"Failed to read config at {path}: {e}") from e

    active = data.get("active_provider")
    if active is not None and active not in PROVIDERS:
        raise ConfigError(f"Invalid active_provider in config: {active!r}")

    providers_raw = data.get("providers", {})
    providers: dict[Provider, ProviderConfig] = {}
    for name, entry in providers_raw.items():
        if name not in PROVIDERS:
            # Unknown provider — skip rather than fail; user may have edited the file.
            continue
        try:
            providers[name] = ProviderConfig(  # type: ignore[arg-type]
                name=name,  # type: ignore[arg-type]
                endpoint=entry["endpoint"],
                api_key=entry["api_key"],
                model=entry["model"],
            )
        except KeyError as e:
            raise ConfigError(
                f"Provider '{name}' in config is missing required field {e}"
            ) from e

    return Config(active_provider=active, providers=providers)


def save_config(cfg: Config) -> Path:
    """Persist config to disk with mode 0600, creating the directory if needed."""
    cdir = config_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    # Make directory user-only too where the OS supports it.
    try:
        os.chmod(cdir, 0o700)
    except OSError:
        pass

    path = config_path()
    payload: dict = {}
    if cfg.active_provider is not None:
        payload["active_provider"] = cfg.active_provider
    if cfg.providers:
        payload["providers"] = {
            name: {
                "endpoint": pc.endpoint,
                "api_key": pc.api_key,
                "model": pc.model,
            }
            for name, pc in cfg.providers.items()
        }

    # Write atomically-ish: write to .tmp then rename, both with restricted perms.
    tmp = path.with_suffix(".toml.tmp")
    with open(tmp, "wb") as f:
        tomli_w.dump(payload, f)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # On Windows chmod is mostly a no-op; nothing more we can do.
        pass
    return path


def warn_if_world_readable() -> None:
    """Print a stderr warning if the config file has loose permissions (POSIX only)."""
    path = config_path()
    if not path.exists() or os.name != "posix":
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:  # any bits set for group or others
        print(
            f"warning: {path} has permissions {oct(mode)}; "
            "consider `chmod 600` to protect API keys.",
            file=sys.stderr,
        )
