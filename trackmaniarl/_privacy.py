"""Narrow content signatures shared by local-artifact privacy gates.

The classifier deliberately reports categories rather than matched values.  It
is a regression guard for a small set of common plaintext credential and
privacy signatures; it is not a general-purpose secret scanner.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_CONTENT_PATTERNS: dict[str, re.Pattern[bytes]] = {
    "personal home path": re.compile(
        rb"(?i)(?:[a-z]:[\\/]+Users[\\/]+[a-z0-9_.-]+|/(?:home|Users)/[a-z0-9_.-]+)"
    ),
    "credential token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
        rb"AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,}|AIza[A-Za-z0-9_-]{35}|"
        rb"wandb_v1_[A-Za-z0-9_]{20,})"
    ),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "URL credentials": re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@"),
}

_FILENAME_PATTERNS = {
    name: pattern
    for name, pattern in _CONTENT_PATTERNS.items()
    if name != "personal home path"
}
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        b"aws_secret_access_key",
        b"gemini_api_key",
        b"gh_token",
        b"github_token",
        b"google_api_key",
        b"hf_token",
        b"huggingface_hub_token",
        b"openai_api_key",
        b"trackmaniarl_distributed_token",
        b"wandb_api_key",
    }
)
_CREDENTIAL_ENV_PATTERN = (
    rb"(?:" + b"|".join(re.escape(name) for name in sorted(_CREDENTIAL_ENV_NAMES)) + rb")"
)
_FILENAME_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?i)" + _CREDENTIAL_ENV_PATTERN + rb"=[^/\\\s]+"
)
_CONFIG_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})
_CONFIG_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?im)(?:^|[\r\n{,])[\t ]*(?:[\"'])?"
    rb"(?P<key>[a-z0-9_.-]+)(?:[\"'])?[\t ]*(?:=|:)[\t ]*(?P<value>[^\r\n]+)"
)
_SAFE_CREDENTIAL_VALUES = frozenset({b"false", b"none", b"null", b"your_token"})
_ENVIRONMENT_REFERENCE = re.compile(
    rb"(?:[\"'])?(?:\$\{[A-Za-z_][A-Za-z0-9_:.-]*\}|\$\{\{[^\r\n}]+\}\})(?:[\"'])?\}?"
)


def content_categories(data: bytes) -> list[str]:
    """Return matched categories only; never include matched content."""
    return [name for name, pattern in _CONTENT_PATTERNS.items() if pattern.search(data)]


def privacy_categories(name: str, data: bytes) -> list[str]:
    """Classify archive members without returning their private names or values."""
    categories = _filename_categories(name)
    categories.extend(content_categories(data))
    if _has_populated_config_credential(name, data):
        categories.append("credential assignment")
    return list(dict.fromkeys(categories))


def _filename_categories(name: str) -> list[str]:
    encoded = name.replace("\\", "/").encode("utf-8", errors="surrogatepass")
    categories = [
        category for category, pattern in _FILENAME_PATTERNS.items() if pattern.search(encoded)
    ]
    if _FILENAME_CREDENTIAL_ASSIGNMENT.search(encoded):
        categories.append("credential assignment")
    return categories


def _has_populated_config_credential(name: str, data: bytes) -> bool:
    if PurePosixPath(name.replace("\\", "/")).suffix.lower() not in _CONFIG_SUFFIXES:
        return False
    for match in _CONFIG_CREDENTIAL_ASSIGNMENT.finditer(data):
        if not _is_credential_key(match.group("key")):
            continue
        raw_value = match.group("value")
        if _is_environment_ref(raw_value):
            continue
        value = _assignment_value(raw_value)
        if value and value.lower() not in _SAFE_CREDENTIAL_VALUES:
            return True
    return False


def _is_credential_key(key: bytes) -> bool:
    normalized = key.lower().replace(b"-", b"_")
    return normalized in _CREDENTIAL_ENV_NAMES


def _assignment_value(value: bytes) -> bytes:
    value = value.split(b"#", maxsplit=1)[0]
    value = value.split(b",", maxsplit=1)[0]
    return value.strip().rstrip(b"}").strip().strip(b"\"'")


def _is_environment_ref(value: bytes) -> bool:
    value = value.split(b"#", maxsplit=1)[0].split(b",", maxsplit=1)[0].strip()
    return _ENVIRONMENT_REFERENCE.fullmatch(value) is not None or re.fullmatch(
        rb"\$[A-Za-z_][A-Za-z0-9_]*", value
    ) is not None
