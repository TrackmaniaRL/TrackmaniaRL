"""Bounded editor-reset keys for an already focused Trackmania on Linux X11."""

from __future__ import annotations

import os
import shutil
import subprocess
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trackmaniarl.trackmania.keyboard_control import ResetGuard


def check_session() -> str:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        raise RuntimeError("Run Trackmania from the local desktop session, not SSH")
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        raise RuntimeError(
            "Trackmania editor reset requires an X11 session; Wayland is not supported"
        )
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Trackmania requires the local graphical X11 session, not headless/SSH")
    tool = shutil.which("xdotool")
    if tool is None:
        raise RuntimeError("Install xdotool in this Linux X11 game session")
    return tool


def _command(tool: str, *args: str) -> str:
    return subprocess.check_output([tool, *args], text=True, timeout=2).strip()


def active_game(tool: str) -> str:
    window = _command(tool, "getactivewindow")
    title = _command(tool, "getwindowname", window)
    if title.strip().casefold() != "trackmania":
        raise RuntimeError("Trackmania must already be the foreground window before reset")
    return window


def tap_reset_key(key: str, *, guard: ResetGuard | None = None) -> None:
    if key not in {"Delete", "Return"}:
        raise ValueError("Only the editor reset/confirmation keys are supported")
    if guard is not None:
        guard.check()
    tool = check_session()
    window = active_game(tool)
    if guard is not None:
        guard.check()
    if _command(tool, "getactivewindow") != window:
        raise RuntimeError("Trackmania foreground changed before reset key")
    # Never activate a window or target a background app; release even on cancellation.
    try:
        subprocess.run([tool, "keydown", key], check=True, timeout=2)
        (sleep if guard is None else guard.wait)(0.1)
    finally:
        subprocess.run([tool, "keyup", key], check=True, timeout=2)
    if guard is not None:
        guard.check()
