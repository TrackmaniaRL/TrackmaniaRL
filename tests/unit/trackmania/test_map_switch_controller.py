from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.trackmania.map_switch_controller import (
    MapSwitchController,
    MapSwitchIndeterminateError,
)
from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trackmaniarl.trackmania import map_switch_controller

    monkeypatch.setattr(map_switch_controller, "_registry_root", lambda: tmp_path / "registry")


class FakePlugin:
    def __init__(self) -> None:
        self.instance = "instance-a"
        self.lease = ""
        self.operation = ""
        self.uid = ""
        self.state = "ready"
        self.fail_ack = False
        self.switch_count = 0

    def __call__(self, request: dict[str, str]) -> dict[str, Any]:
        command = request["command"]
        if command == "get_instance":
            return {"status": "ok", "protocol_version": "3", "instance_epoch": self.instance}
        assert request["instance_epoch"] == self.instance
        if command == "claim":
            self.lease = request["epoch"]
        if command == "switch_map":
            self.operation = request["operation_id"]
            self.uid = request["expected_map_uid"]
            self.switch_count += 1
            if self.fail_ack:
                raise TimeoutError("lost acknowledgement")
        return {
            "status": "ok",
            "protocol_version": "3",
            "instance_epoch": self.instance,
            "epoch": self.lease,
            "operation_id": self.operation,
            "state": self.state,
            "map_uid": self.uid,
            "ready": "true",
        }


@pytest.mark.parametrize("replacement", ["in_place", "rename"])
def test_rolled_back_journal_cannot_erase_pending_switch(tmp_path: Path, replacement: str) -> None:
    import shutil

    map_path = tmp_path / "map.gbx"
    map_path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(db, "instance", create=True)
    old = tmp_path / "old.sqlite"
    shutil.copyfile(db, old)
    plugin = FakePlugin()
    plugin.fail_ack = True
    controller = MapSwitchController(journal, plugin)
    with pytest.raises(MapSwitchIndeterminateError):
        controller.switch(map_path, "uid")
    assert journal.pending() is not None
    if replacement == "in_place":
        shutil.copyfile(old, db)
    else:
        old.replace(db)
    assert journal.pending() is None
    plugin.instance = "restarted"
    with pytest.raises(MapSwitchIndeterminateError, match="unresolved"):
        controller.switch(map_path, "uid")
    with pytest.raises(MapSwitchIndeterminateError, match="disagree"):
        controller.recover()
    controller.close()
    with (
        MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened,
        pytest.raises(MapSwitchIndeterminateError, match="unresolved"),
    ):
        reopened.switch(map_path, "uid")
    assert plugin.switch_count == 1


def test_missing_or_corrupt_quarantine_marker_never_recovers_pending(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    plugin.fail_ack = True
    controller = MapSwitchController(journal, plugin)
    with pytest.raises(MapSwitchIndeterminateError):
        controller.switch(path, "uid")
    marker = controller._quarantine_path
    marker.unlink()
    with pytest.raises(MapSwitchIndeterminateError, match="disagree"):
        controller.recover()
    marker.write_bytes(b"invalid")
    with pytest.raises(MapSwitchIndeterminateError, match="invalid"):
        controller.recover()
    controller.close()
    assert plugin.switch_count == 1


def test_terminal_marker_survives_failed_cleanup_and_is_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(db, "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    clear = controller._clear_quarantine
    monkeypatch.setattr(controller, "_clear_quarantine", lambda _: (_ for _ in ()).throw(OSError()))
    with pytest.raises(MapSwitchIndeterminateError):
        controller.switch(path, "uid")
    assert journal.latest()[1] == "terminal"  # type: ignore[index]
    assert controller._read_quarantine() is not None
    monkeypatch.setattr(controller, "_clear_quarantine", clear)
    controller.close()
    with MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened:
        reopened.recover()
        assert reopened._read_quarantine() is None
    assert plugin.switch_count == 1


def test_failure_after_marker_before_journal_prepare_stays_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    monkeypatch.setattr(journal, "prepare", lambda _: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        controller.switch(path, "uid")
    assert controller._read_quarantine() is not None
    assert journal.pending() is None
    with pytest.raises(MapSwitchIndeterminateError, match="disagree"):
        controller.recover()
    with pytest.raises(MapSwitchIndeterminateError, match="unresolved"):
        controller.switch(path, "uid")
    controller.close()
    assert plugin.switch_count == 0


def test_failure_syncing_marker_prevents_any_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania import map_switch_controller as module

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    monkeypatch.setattr(
        module, "_sync_registry_directory", lambda _: (_ for _ in ()).throw(OSError("sync failed"))
    )
    with pytest.raises(OSError, match="sync failed"):
        controller.switch(path, "uid")
    assert controller._read_quarantine() is not None
    assert journal.pending() is None
    assert plugin.switch_count == 0
    controller.close()


def test_failure_after_terminal_marker_removal_does_not_repeat_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania import map_switch_controller as module

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    sync = module._sync_registry_directory
    calls = 0

    def fail_second_sync(root: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        sync(root)

    monkeypatch.setattr(module, "_sync_registry_directory", fail_second_sync)
    with pytest.raises(MapSwitchIndeterminateError):
        controller.switch(path, "uid")
    assert controller._read_quarantine() is None
    latest = journal.latest()
    assert latest is not None
    assert latest[1] == "terminal"
    controller.verify_active_map("uid")
    controller.close()
    assert plugin.switch_count == 1


def test_switch_saves_terminal_and_reuses_lease_after_controller_restart(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(db, "instance", create=True)
    plugin = FakePlugin()
    with MapSwitchController(journal, plugin) as controller:
        controller.switch(path, "uid")
    latest = journal.latest()
    assert latest is not None
    assert latest[1] == "terminal"
    journal = MapSwitchJournal(db, "instance")
    with MapSwitchController(journal, plugin) as controller:
        controller.switch(path, "uid")
    assert plugin.switch_count == 2
    assert journal.pending() is None


def test_lost_ack_is_quarantined_then_same_operation_can_be_polled(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    plugin.fail_ack = True
    controller = MapSwitchController(MapSwitchJournal(db, "instance", create=True), plugin)
    with pytest.raises(MapSwitchIndeterminateError, match="quarantined"):
        controller.switch(path, "uid")
    controller.close()
    journal = MapSwitchJournal(db, "instance")
    assert journal.pending() is not None
    with MapSwitchController(journal, plugin) as controller:
        with pytest.raises(MapSwitchIndeterminateError, match="unresolved"):
            controller.switch(path, "uid")
        controller.recover()
    assert plugin.switch_count == 1
    assert journal.pending() is None


def test_plugin_restart_after_lost_ack_never_dispatches_or_resolves(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    plugin.fail_ack = True
    controller = MapSwitchController(journal, plugin)
    with pytest.raises(MapSwitchIndeterminateError):
        controller.switch(path, "uid")
    plugin.instance = "instance-b"
    with pytest.raises(MapSwitchIndeterminateError, match="epoch changed"):
        controller.recover()
    controller.close()
    assert journal.pending() is not None
    assert plugin.switch_count == 1


def test_mismatched_active_map_never_resolves(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()

    def wrong_uid(request: dict[str, str]) -> dict[str, Any]:
        result = plugin(request)
        if request["command"] == "switch_map":
            result["map_uid"] = "different"
        return result

    with (
        MapSwitchController(journal, wrong_uid) as controller,
        pytest.raises(MapSwitchIndeterminateError, match="matching operation"),
    ):
        controller.switch(path, "uid")
    assert journal.pending() is not None


@pytest.mark.parametrize("state", ["accepted", "loading"])
def test_nonterminal_timeout_keeps_quarantine(tmp_path: Path, state: str) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    plugin.state = state
    with (
        MapSwitchController(journal, plugin) as controller,
        pytest.raises(MapSwitchIndeterminateError, match="terminal"),
    ):
        controller.switch(path, "uid", timeout_s=0.001)
    assert journal.pending() is not None
    assert plugin.switch_count == 1


@pytest.mark.parametrize("fault", ["epoch", "uid", "missing", "file"])
def test_recheck_after_initial_ready_fails_closed(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()

    def change(request: dict[str, str]) -> dict[str, Any]:
        result = plugin(request)
        if request["command"] == "switch_map" and fault == "file":
            path.write_bytes(b"changed")
        if request["command"] == "switch_status":
            if fault == "epoch":
                result["instance_epoch"] = "different"
            elif fault == "uid":
                result["map_uid"] = "different"
            elif fault == "missing":
                del result["operation_id"]
        return result

    with (
        MapSwitchController(journal, change) as controller,
        pytest.raises(MapSwitchIndeterminateError),
    ):
        controller.switch(path, "uid")
    assert journal.pending() is not None
    assert plugin.switch_count == 1


@pytest.mark.parametrize(
    ("command", "expected"), [("claim", "prepared"), ("switch_map", "dispatching")]
)
def test_interruption_does_not_erase_intent(tmp_path: Path, command: str, expected: str) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()

    def interrupt(request: dict[str, str]) -> dict[str, Any]:
        if request["command"] == command:
            raise KeyboardInterrupt
        return plugin(request)

    with MapSwitchController(journal, interrupt) as controller, pytest.raises(KeyboardInterrupt):
        controller.switch(path, "uid")
    pending = journal.pending()
    assert pending is not None
    assert pending[1] == expected
    assert plugin.switch_count == 0


def test_reconfirm_training_map_refuses_plugin_restart(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    controller.switch(path, "uid")
    controller.verify_active_map("uid")
    plugin.instance = "new-instance"
    with pytest.raises(MapSwitchIndeterminateError, match="plugin restarted"):
        controller.verify_active_map("uid")
    controller.close()


def test_instance_lock_spans_verification_and_collection(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal_path = tmp_path / "switch.sqlite"
    journal = MapSwitchJournal(journal_path, "instance", create=True)
    plugin = FakePlugin()
    controller = MapSwitchController(journal, plugin)
    controller.switch(path, "uid")
    controller.verify_active_map("uid")
    script = (
        "import sys; from pathlib import Path; "
        "from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal; "
        "from trackmaniarl.trackmania import map_switch_controller as module; "
        "from trackmaniarl.trackmania.map_switch_controller import MapSwitchController; "
        "module._registry_root=lambda: Path(sys.argv[2]); "
        "c=MapSwitchController(MapSwitchJournal(Path(sys.argv[1]), 'instance'), lambda _: {}); "
        "c.close()"
    )
    held = subprocess.run(
        [sys.executable, "-c", script, str(journal_path), str(tmp_path / "registry")],
        capture_output=True,
        check=False,
    )
    assert held.returncode != 0
    assert b"game instance is already owned" in held.stderr
    controller.close()
    released = subprocess.run(
        [sys.executable, "-c", script, str(journal_path), str(tmp_path / "registry")],
        capture_output=True,
        check=False,
    )
    assert released.returncode == 0, released.stderr.decode(errors="replace")
    with pytest.raises(MapSwitchIndeterminateError, match="ownership was released"):
        controller.verify_active_map("uid")


def test_second_journal_cannot_forget_pending_after_plugin_restart(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    first = MapSwitchJournal(tmp_path / "first.sqlite", "same-instance", create=True)
    second = MapSwitchJournal(tmp_path / "second.sqlite", "same-instance", create=True)
    plugin = FakePlugin()
    plugin.fail_ack = True
    with MapSwitchController(first, plugin) as controller:
        with pytest.raises(MapSwitchIndeterminateError, match="already owned"):
            MapSwitchController(second, plugin)
        with pytest.raises(MapSwitchIndeterminateError):
            controller.switch(path, "uid")
    assert first.pending() is not None
    plugin.instance = "after-restart"
    with pytest.raises(MapSwitchIndeterminateError, match="bound to a different journal"):
        MapSwitchController(second, plugin)
    assert first.pending() is not None
    assert plugin.switch_count == 1


def test_instance_lock_refuses_hardlink_without_modifying_target(tmp_path: Path) -> None:
    import os

    from trackmaniarl.trackmania.map_switch_controller import _InstanceLock

    victim = tmp_path / "victim"
    victim.write_bytes(b"")
    link = tmp_path / "lock"
    os.link(victim, link)
    with pytest.raises(OSError, match="unsafe"):
        _InstanceLock(link)
    assert victim.read_bytes() == b""


def test_endpoint_binding_survives_instance_label_change_and_plugin_restart(tmp_path: Path) -> None:
    from trackmaniarl.trackmania.map_switch_controller import OpenPlanetSwitchTransport

    class EndpointPlugin(OpenPlanetSwitchTransport, FakePlugin):
        def __init__(self) -> None:
            OpenPlanetSwitchTransport.__init__(self, "127.0.0.1", 9001)
            FakePlugin.__init__(self)

        def __call__(self, request: dict[str, str]) -> dict[str, Any]:
            return FakePlugin.__call__(self, request)

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    first = MapSwitchJournal(tmp_path / "first.sqlite", "first-label", create=True)
    second = MapSwitchJournal(tmp_path / "second.sqlite", "different-label", create=True)
    plugin = EndpointPlugin()
    plugin.fail_ack = True
    with (
        MapSwitchController(first, plugin) as controller,
        pytest.raises(MapSwitchIndeterminateError),
    ):
        controller.switch(path, "uid")
    plugin.instance = "restarted"
    with pytest.raises(MapSwitchIndeterminateError, match="bound to a different journal"):
        MapSwitchController(second, plugin)
    assert first.pending() is not None
    assert plugin.switch_count == 1


def test_registry_refuses_hardlink_record(tmp_path: Path) -> None:
    import hashlib
    import os

    registry = tmp_path / "registry"
    registry.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"private")
    key = hashlib.sha256(b"instance:instance").hexdigest()
    os.link(victim, registry / f"{key}.json")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with pytest.raises(MapSwitchIndeterminateError, match="unsafe game instance registry"):
        MapSwitchController(journal, FakePlugin())
    assert victim.read_bytes() == b"private"
