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
            "remaining_operations": 4096 - self.switch_count,
        }


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        *(
            (phase, f"switch did not reach a verified terminal state (phase: {phase})")
            for phase in ("accepted", "leaving", "loading", "ready")
        ),
        (None, "switch did not reach a verified terminal state"),
        ([], "switch did not reach a verified terminal state"),
        ("unexpected plugin text", "switch did not reach a verified terminal state"),
    ],
)
def test_indeterminate_phase_is_reported_without_clearing_quarantine(
    tmp_path: Path, phase: object, expected: str
) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)

    def exchange(request: dict[str, str]) -> dict[str, Any]:
        response = plugin(request)
        if request["command"] in {"switch_map", "switch_status"}:
            response["state"] = "indeterminate"
            if phase is not None:  # Old protocol-3 implementations omit this field.
                response["failure_phase"] = phase
        return response

    with MapSwitchController(journal, exchange) as controller:
        for operation in (lambda: controller.switch(path, "uid"), controller.recover):
            with pytest.raises(MapSwitchIndeterminateError) as caught:
                operation()
            assert str(caught.value) == expected
            assert journal.pending() is not None
            assert controller._quarantine_path.exists()
    assert plugin.switch_count == 1


def test_telemetry_guard_blocks_restart_even_when_plugin_reports_ready(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    commands: list[str] = []

    def exchange(request: dict[str, str]) -> dict[str, Any]:
        commands.append(request["command"])
        return plugin(request)

    with MapSwitchController(MapSwitchJournal(db, "instance", create=True), exchange) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        owner.switch(path, "evaluation")
        owner.verify_active_map("evaluation")
        marker = owner._telemetry_quarantine_path
        assert marker.exists()
    # Simulate a dead owner: even a perfectly ready plugin cannot resolve the guard.
    assert plugin.state == "ready"
    before = list(commands)
    with MapSwitchController(MapSwitchJournal(db, "instance"), exchange) as reopened:
        for operation in (
            lambda: reopened.switch(path, "training"),
            lambda: reopened.verify_active_map("evaluation"),
            lambda: reopened.telemetry_identity("evaluation"),
            lambda: reopened.require_capacity("evaluation", 1),
            reopened.recover,
            reopened.begin_evaluation_guard,
            reopened.finish_evaluation_guard,
        ):
            with pytest.raises(MapSwitchIndeterminateError):
                operation()
        assert marker.exists()
    assert commands == before
    assert plugin.switch_count == 2


def test_armed_guard_survives_abrupt_process_exit(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    registry = tmp_path / "registry"
    script = (
        "import os,sys; from pathlib import Path; "
        "from tests.unit.trackmania.test_map_switch_controller import FakePlugin; "
        "from trackmaniarl.trackmania import map_switch_controller as module; "
        "from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal; "
        "module._registry_root=lambda: Path(sys.argv[2]); "
        "controller=module.MapSwitchController("
        "MapSwitchJournal(Path(sys.argv[1]), 'instance', create=True), FakePlugin()); "
        "controller.switch(Path(sys.argv[3]), 'training'); "
        "controller.begin_evaluation_guard(); os._exit(23)"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(db), str(registry), str(path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 23, child.stderr.decode(errors="replace")
    commands: list[str] = []
    plugin = FakePlugin()

    def exchange(request: dict[str, str]) -> dict[str, Any]:
        commands.append(request["command"])
        return plugin(request)

    with MapSwitchController(MapSwitchJournal(db, "instance"), exchange) as reopened:
        assert reopened._telemetry_quarantine_path.exists()
        with pytest.raises(MapSwitchIndeterminateError, match="manual recovery"):
            reopened.switch(path, "training")
        with pytest.raises(MapSwitchIndeterminateError, match="manual recovery"):
            reopened.recover()
        with pytest.raises(MapSwitchIndeterminateError, match="manual recovery"):
            reopened.verify_active_map("training")
    assert commands == []
    assert plugin.switch_count == 0


def test_guard_requires_restored_training_and_explicit_owner_finish(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, plugin) as controller:
        controller.switch(path, "training")
        controller.begin_evaluation_guard()
        with pytest.raises(MapSwitchIndeterminateError, match="already active"):
            controller.begin_evaluation_guard()
        controller.switch(path, "evaluation")
        assert controller._telemetry_quarantine_path.exists()
        controller.switch(path, "training")
        controller.finish_evaluation_guard()
        assert not controller._telemetry_quarantine_path.exists()
        controller.verify_active_map("training")


def test_guard_finish_rejects_wrong_map_and_poison_owner(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, plugin) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        owner.switch(path, "evaluation")
        with pytest.raises(MapSwitchIndeterminateError, match="matching terminal"):
            owner.finish_evaluation_guard()
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.switch(path, "training")
        assert owner._telemetry_quarantine_path.exists()


def test_guard_finish_does_not_require_player_ready(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    player_ready = True

    def exchange(request: dict[str, str]) -> dict[str, Any]:
        response = plugin(request)
        if request["command"] == "switch_status" and not player_ready:
            response["ready"] = "false"
        return response

    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, exchange) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        player_ready = False
        owner.finish_evaluation_guard()
        assert not owner._telemetry_quarantine_path.exists()


def test_telemetry_failure_poison_is_idempotent_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    with MapSwitchController(MapSwitchJournal(db, "instance", create=True), plugin) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        owner.quarantine_uncertain_telemetry()
        owner.quarantine_uncertain_telemetry()
        for operation in (
            lambda: owner.switch(path, "training"),
            lambda: owner.verify_active_map("training"),
            owner.recover,
            owner.finish_evaluation_guard,
        ):
            with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
                operation()
        assert owner._telemetry_quarantine_path.exists()
    with (
        MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened,
        pytest.raises(MapSwitchIndeterminateError, match="manual recovery"),
    ):
        reopened.recover()
    assert plugin.switch_count == 1


def test_standalone_telemetry_quarantine_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    with MapSwitchController(MapSwitchJournal(db, "instance", create=True), plugin) as owner:
        owner.quarantine_uncertain_telemetry()
        owner.quarantine_uncertain_telemetry()
        assert owner._telemetry_quarantine_path.exists()
    with (
        MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened,
        pytest.raises(MapSwitchIndeterminateError, match="manual recovery"),
    ):
        reopened.recover()


def test_guard_sync_failure_blocks_the_current_and_next_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania import map_switch_controller as module

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    with MapSwitchController(MapSwitchJournal(db, "instance", create=True), plugin) as owner:
        owner.switch(path, "training")
        with monkeypatch.context() as fail_sync:
            fail_sync.setattr(
                module, "_sync_registry_directory", lambda _: (_ for _ in ()).throw(OSError("sync"))
            )
            with pytest.raises(OSError, match="sync"):
                owner.begin_evaluation_guard()
        assert owner._telemetry_quarantine_path.exists()
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.switch(path, "evaluation")
    with (
        MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened,
        pytest.raises(MapSwitchIndeterminateError, match="manual recovery"),
    ):
        reopened.recover()
    assert plugin.switch_count == 1


def test_guard_creation_failure_blocks_current_owner_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, plugin) as owner:
        owner.switch(path, "training")
        original_open = os.open

        def fail_marker(target: Any, *args: Any, **kwargs: Any) -> int:
            if target == owner._telemetry_quarantine_path:
                raise OSError("marker creation failed")
            return original_open(target, *args, **kwargs)

        with monkeypatch.context() as failure:
            failure.setattr(os, "open", fail_marker)
            with pytest.raises(OSError, match="marker creation failed"):
                owner.begin_evaluation_guard()
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.switch(path, "evaluation")
    assert plugin.switch_count == 1


def test_guard_disappearance_poison_current_owner(tmp_path: Path) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, plugin) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        owner._telemetry_quarantine_path.unlink()
        with pytest.raises(MapSwitchIndeterminateError, match="marker disappeared"):
            owner.switch(path, "evaluation")
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.recover()
    assert plugin.switch_count == 1


def test_guard_finish_sync_failure_recreates_marker_and_blocks_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania import map_switch_controller as module

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    db = tmp_path / "switch.sqlite"
    plugin = FakePlugin()
    with MapSwitchController(MapSwitchJournal(db, "instance", create=True), plugin) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        with monkeypatch.context() as failure:
            failure.setattr(
                module, "_sync_registry_directory", lambda _: (_ for _ in ()).throw(OSError("sync"))
            )
            with pytest.raises(OSError, match="sync"):
                owner.finish_evaluation_guard()
        assert owner._telemetry_quarantine_path.exists()
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.switch(path, "training")
    with (
        MapSwitchController(MapSwitchJournal(db, "instance"), plugin) as reopened,
        pytest.raises(MapSwitchIndeterminateError, match="manual recovery"),
    ):
        reopened.recover()
    assert plugin.switch_count == 1


def test_guard_finish_unlink_failure_blocks_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    plugin = FakePlugin()
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    with MapSwitchController(journal, plugin) as owner:
        owner.switch(path, "training")
        owner.begin_evaluation_guard()
        original_unlink = Path.unlink

        def fail_marker(target: Path, *args: Any, **kwargs: Any) -> None:
            if target == owner._telemetry_quarantine_path:
                raise OSError("unlink failed")
            original_unlink(target, *args, **kwargs)

        with monkeypatch.context() as failure:
            failure.setattr(Path, "unlink", fail_marker)
            with pytest.raises(OSError, match="unlink failed"):
                owner.finish_evaluation_guard()
        assert owner._telemetry_quarantine_path.exists()
        with pytest.raises(MapSwitchIndeterminateError, match="uncertain telemetry"):
            owner.switch(path, "training")


@pytest.mark.parametrize("remaining", [None, True, -1, 4097, "3"])
def test_capacity_rejects_missing_or_invalid_report_without_dispatch(
    tmp_path: Path, remaining: Any
) -> None:
    class CapacityPlugin(FakePlugin):
        def __call__(self, request: dict[str, str]) -> dict[str, Any]:
            response = super().__call__(request)
            if request["command"] == "switch_status":
                response["remaining_operations"] = remaining
            return response

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = CapacityPlugin()
    with MapSwitchController(journal, plugin) as controller:
        controller.switch(path, "uid")
        with pytest.raises(MapSwitchIndeterminateError, match="valid capacity"):
            controller.require_capacity("uid", 3)
        assert journal.pending() is None
        assert plugin.switch_count == 1


def test_capacity_requires_room_for_suite_and_restore(tmp_path: Path) -> None:
    class CapacityPlugin(FakePlugin):
        remaining = 2

        def __call__(self, request: dict[str, str]) -> dict[str, Any]:
            response = super().__call__(request)
            if request["command"] == "switch_status":
                response["remaining_operations"] = self.remaining
            return response

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = CapacityPlugin()
    with MapSwitchController(journal, plugin) as controller:
        controller.switch(path, "uid")
        with pytest.raises(MapSwitchIndeterminateError, match="insufficient switch capacity"):
            controller.require_capacity("uid", 3)
        assert journal.pending() is None
        assert plugin.switch_count == 1
        plugin.remaining = 3
        controller.require_capacity("uid", 3)


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


def test_original_map_reappearing_during_failed_load_does_not_resolve_or_redispatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.gbx"
    path.write_bytes(b"target")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    status_calls = 0
    status_operations: list[str] = []

    def exchange(request: dict[str, str]) -> dict[str, Any]:
        nonlocal status_calls
        response = plugin(request)
        if request["command"] == "switch_map":
            response.update(state="accepted", map_uid="original", ready="true")
        if request["command"] == "switch_status":
            status_calls += 1
            status_operations.append(request["operation_id"])
            if status_calls == 1:
                response.update(state="loading", map_uid="", ready="false")
            elif status_calls == 2:
                response.update(state="loading", map_uid="original", ready="false")
            else:
                response.update(
                    state="indeterminate",
                    failure_phase="loading",
                    map_uid="original",
                    ready="true",
                )
        return response

    with MapSwitchController(journal, exchange) as controller:
        with pytest.raises(MapSwitchIndeterminateError, match="phase: loading"):
            controller.switch(path, "target")
        pending = journal.pending()
        assert pending is not None
        assert pending[1] == "dispatching"
        assert controller._quarantine_path.exists()
        with pytest.raises(MapSwitchIndeterminateError, match="phase: loading"):
            controller.recover()
        assert journal.pending() == pending
        assert controller._quarantine_path.exists()
    assert plugin.switch_count == 1
    assert status_operations == [pending[0].operation_id] * 4


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


def test_default_timeout_allows_both_plugin_switch_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania import map_switch_controller

    path = tmp_path / "map.gbx"
    path.write_bytes(b"map")
    journal = MapSwitchJournal(tmp_path / "switch.sqlite", "instance", create=True)
    plugin = FakePlugin()
    clock = [0.0]
    monkeypatch.setattr(map_switch_controller, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        map_switch_controller, "sleep", lambda _: clock.__setitem__(0, clock[0] + 5)
    )

    def switch_over_two_phases(request: dict[str, str]) -> dict[str, Any]:
        response = plugin(request)
        if request["command"] in ("switch_map", "switch_status"):
            response["state"] = "leaving" if clock[0] < 25 else "loading"
            if clock[0] >= 45:
                response["state"] = "ready"
        return response

    with MapSwitchController(journal, switch_over_two_phases) as controller:
        controller.switch(path, "uid")
    assert clock[0] == 45
    assert journal.latest()[1] == "terminal"  # type: ignore[index]
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
