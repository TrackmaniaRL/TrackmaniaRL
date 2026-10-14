"""No real keyboard input: X11 focus, session and key-release regression guards."""

from types import SimpleNamespace

import pytest

from trackmaniarl.trackmania import keyboard_control, linux_reset


@pytest.fixture
def x11(monkeypatch: pytest.MonkeyPatch) -> list:
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(linux_reset.shutil, "which", lambda _: "/usr/bin/xdotool")
    monkeypatch.setattr(linux_reset, "sleep", lambda _: None)
    calls = []
    monkeypatch.setattr(linux_reset.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(
        linux_reset,
        "_command",
        lambda tool, *args: "Trackmania" if args[0] == "getwindowname" else "123",
    )
    return calls


def test_wayland_never_sends_keys(x11: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    with pytest.raises(RuntimeError, match="Wayland"):
        linux_reset.tap_reset_key("Delete")
    assert not x11


def test_headless_never_sends_keys(x11: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(RuntimeError, match="headless"):
        linux_reset.tap_reset_key("Return")
    assert not x11


def test_wrong_foreground_never_sends_keys(x11: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linux_reset,
        "_command",
        lambda tool, *args: "Terminal" if args[0] == "getwindowname" else "123",
    )
    with pytest.raises(RuntimeError, match="foreground"):
        linux_reset.tap_reset_key("Delete")
    assert not x11


def test_focus_change_never_sends_keys(x11: list, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(["123", "Trackmania", "456"])
    monkeypatch.setattr(linux_reset, "_command", lambda *args: next(replies))
    with pytest.raises(RuntimeError, match="changed"):
        linux_reset.tap_reset_key("Delete")
    assert not x11


def test_cancelled_wait_releases_key(x11: list) -> None:
    def cancel(_: float) -> None:
        raise RuntimeError("STOP")

    guard = SimpleNamespace(check=lambda: None, wait=cancel)
    with pytest.raises(RuntimeError, match="STOP"):
        linux_reset.tap_reset_key("Delete", guard=guard)
    assert x11 == [
        ["/usr/bin/xdotool", "keydown", "Delete"],
        ["/usr/bin/xdotool", "keyup", "Delete"],
    ]


def test_linux_editor_reset_keeps_delete_wait_return_order(
    x11: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keyboard_control, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(keyboard_control, "sleep", lambda _: None)
    keyboard_control.restart_trackmania_editor_validation()
    assert [call[1:] for call in x11] == [
        ["keydown", "Delete"],
        ["keyup", "Delete"],
        ["keydown", "Return"],
        ["keyup", "Return"],
    ]


def test_forwarded_ssh_never_sends_keys(x11: list, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSH_CONNECTION", "forwarded")
    with pytest.raises(RuntimeError, match="SSH"):
        linux_reset.tap_reset_key("Delete")
    assert not x11
