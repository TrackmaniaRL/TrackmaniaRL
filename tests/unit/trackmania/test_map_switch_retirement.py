from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pytest

from tests.unit.trackmania.test_map_switch_controller import FakePlugin
from trackmaniarl.trackmania import map_switch_controller, map_switch_retirement
from trackmaniarl.trackmania.map_switch_controller import (
    MapSwitchController,
    MapSwitchIndeterminateError,
    OpenPlanetSwitchTransport,
    _InstanceLock,
)
from trackmaniarl.trackmania.map_switch_journal import MapSwitchIntent, MapSwitchJournal
from trackmaniarl.trackmania.map_switch_retirement import RetirementAttestation

pytestmark = pytest.mark.skipif(os.name != "nt", reason="retirement requires Windows file locks")
_BEFORE_SENTINEL = (
    "archive_created",
    "archived:binding.json",
    "archived:pending",
    "archived:old-journal.sqlite",
    "archived:operator-evidence",
    "manifest_written",
    "archive_synced",
)
_AFTER_SENTINEL = ("sentinel_replaced", "sentinel_installed", "pending_unlinked", "pending_retired")


class _FakeEndpoint(OpenPlanetSwitchTransport, FakePlugin):
    def __init__(self) -> None:
        OpenPlanetSwitchTransport.__init__(self, "127.0.0.1", 9001)
        FakePlugin.__init__(self)
        self.fail_ack = True

    def __call__(self, request: dict[str, str]) -> dict[str, Any]:
        return FakePlugin.__call__(self, request)


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "registry"
    monkeypatch.setattr(map_switch_controller, "_registry_root", lambda: root)
    monkeypatch.setattr(map_switch_retirement, "_registry_root", lambda: root)
    # These tests never bind ports or access a running game.
    monkeypatch.setattr(map_switch_retirement, "_reserve_game_ports", nullcontext)
    old_path, new_path = tmp_path / "old.sqlite", tmp_path / "new.sqlite"
    old = MapSwitchJournal(old_path, "old", create=True)
    new = MapSwitchJournal(new_path, "new", create=True)
    target = tmp_path / "map.gbx"
    target.write_bytes(b"map")
    endpoint = _FakeEndpoint()
    with (
        MapSwitchController(old, endpoint) as controller,
        pytest.raises(MapSwitchIndeterminateError),
    ):
        controller.switch(target, "uid")
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Operator's separately checked process and ownership evidence", "utf-8")
    attestation = RetirementAttestation(True, True, True, True, evidence)
    key = hashlib.sha256(b"endpoint:127.0.0.1:9001").hexdigest()
    return {
        "root": root,
        "key": key,
        "old": old,
        "new": new,
        "endpoint": endpoint,
        "attestation": attestation,
        "old_bytes": old_path.read_bytes(),
    }


def _retire(item: dict[str, Any]) -> Path:
    return map_switch_retirement.retire_quarantined_endpoint(
        old_journal=item["old"].path,
        old_instance_id="old",
        new_journal=item["new"].path,
        new_instance_id="new",
        attestation=item["attestation"],
    )


def test_retirement_preserves_dispatch_and_archives_exact_evidence(
    isolated: dict[str, Any],
) -> None:
    root, key = isolated["root"], isolated["key"]
    old_binding = (root / (key + ".json")).read_bytes()
    marker = (root / (key + ".pending")).read_bytes()
    archive = _retire(isolated)
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]
    assert isolated["old"].pending()[1] == "dispatching"
    assert (archive / "old-journal.sqlite").read_bytes() == isolated["old_bytes"]
    assert (archive / "binding.json").read_bytes() == old_binding
    assert (archive / "pending").read_bytes() == marker
    manifest = json.loads((archive / "manifest.json").read_text("utf-8"))
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == digest
    assert not (root / (key + ".pending")).exists()
    with MapSwitchController(isolated["new"], isolated["endpoint"]):
        pass
    with pytest.raises(MapSwitchIndeterminateError, match="different journal"):
        MapSwitchController(isolated["old"], isolated["endpoint"])
    assert isolated["endpoint"].switch_count == 1


@pytest.mark.parametrize("alias_target", ["old", "new"])
def test_windows_short_alias_publishes_canonical_fresh_journal(
    isolated: dict[str, Any], alias_target: str
) -> None:
    import ctypes
    from ctypes import wintypes

    path = isolated["root"].parent / "a-very-long-fresh-switch-journal-name.sqlite"
    journal = MapSwitchJournal(path, "new", create=True)
    alias_path = isolated["old"].path if alias_target == "old" else path
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    short_name = kernel.GetShortPathNameW
    short_name.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    short_name.restype = wintypes.DWORD
    length = short_name(str(alias_path), None, 0)
    if not length:
        pytest.skip("the volume does not provide a short-name alias")
    buffer = ctypes.create_unicode_buffer(length)
    if not short_name(str(alias_path), buffer, length):
        pytest.skip("the volume does not provide a short-name alias")
    alias = Path(buffer.value)
    if os.path.normcase(str(alias)) == os.path.normcase(str(alias_path.resolve())):
        pytest.skip("the volume has no distinct 8.3 alias for this path")
    assert os.path.samefile(alias, alias_path)
    map_switch_retirement.retire_quarantined_endpoint(
        old_journal=alias if alias_target == "old" else isolated["old"].path,
        old_instance_id="old",
        new_journal=alias if alias_target == "new" else path,
        new_instance_id="new",
        attestation=isolated["attestation"],
    )
    # Constructors use MapSwitchJournal.path (Path.resolve), not the input alias.
    with MapSwitchController(journal, isolated["endpoint"]):
        pass
    assert isolated["old"].pending()[1] == "dispatching"


@pytest.mark.parametrize("phase", [*_BEFORE_SENTINEL, *_AFTER_SENTINEL])
def test_failure_at_each_precommit_boundary_cannot_dispatch(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    def fail(name: str) -> None:
        if name == phase:
            raise OSError("injected interruption")

    monkeypatch.setattr(map_switch_retirement, "_phase", fail)
    with pytest.raises(OSError, match="injected"):
        _retire(isolated)
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]
    assert isolated["old"].pending()[1] == "dispatching"
    with pytest.raises(MapSwitchIndeterminateError):
        MapSwitchController(isolated["new"], isolated["endpoint"])
    if phase in _BEFORE_SENTINEL:
        with (
            MapSwitchController(isolated["old"], isolated["endpoint"]) as owner,
            pytest.raises(MapSwitchIndeterminateError, match="unresolved"),
        ):
            owner.switch(isolated["old"].path, "uid")
    else:
        with pytest.raises(MapSwitchIndeterminateError, match="different journal"):
            MapSwitchController(isolated["old"], isolated["endpoint"])
    assert isolated["endpoint"].switch_count == 1


def _exit_retirement_process(isolated: dict[str, Any], phase: str) -> None:
    script = r"""
import os, sys
from contextlib import nullcontext
from pathlib import Path
from trackmaniarl.trackmania import map_switch_retirement as module
module._registry_root = lambda: Path(sys.argv[1])
module._reserve_game_ports = nullcontext
module._phase = lambda name: os._exit(37) if name == sys.argv[5] else None
module.retire_quarantined_endpoint(
    old_journal=Path(sys.argv[2]), old_instance_id="old",
    new_journal=Path(sys.argv[3]), new_instance_id="new",
    attestation=module.RetirementAttestation(True, True, True, True, Path(sys.argv[4])),
)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(isolated["root"]),
            str(isolated["old"].path),
            str(isolated["new"].path),
            str(isolated["attestation"].evidence_path),
            phase,
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 37, result.stderr.decode(errors="replace")


@pytest.mark.parametrize("phase", [*_BEFORE_SENTINEL, *_AFTER_SENTINEL])
def test_process_exit_at_each_precommit_boundary_remains_blocked(
    isolated: dict[str, Any], phase: str
) -> None:
    _exit_retirement_process(isolated, phase)
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]
    assert isolated["old"].pending()[1] == "dispatching"
    with pytest.raises(MapSwitchIndeterminateError):
        MapSwitchController(isolated["new"], isolated["endpoint"])
    if phase in _AFTER_SENTINEL:
        with pytest.raises(MapSwitchIndeterminateError):
            MapSwitchController(isolated["old"], isolated["endpoint"])
    else:
        with (
            MapSwitchController(isolated["old"], isolated["endpoint"]) as owner,
            pytest.raises(MapSwitchIndeterminateError, match="unresolved"),
        ):
            owner.switch(isolated["old"].path, "uid")


def test_process_exit_at_commit_preserves_old_evidence(isolated: dict[str, Any]) -> None:
    _exit_retirement_process(isolated, "new_binding_replaced")
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]
    assert isolated["old"].pending()[1] == "dispatching"
    archives = list(isolated["root"].glob("retired-*"))
    assert len(archives) == 1
    assert (archives[0] / "old-journal.sqlite").read_bytes() == isolated["old_bytes"]
    # Publication is the commit point; after it, the fresh binding may be live.
    with MapSwitchController(isolated["new"], isolated["endpoint"]):
        pass


@pytest.mark.parametrize(
    "fault", ["marker", "binding", "telemetry", "sidecar", "writer", "new_used"]
)
def test_inconsistent_or_busy_inputs_leave_existing_quarantine(
    isolated: dict[str, Any], fault: str
) -> None:
    root, key = isolated["root"], isolated["key"]
    marker, binding = root / (key + ".pending"), root / (key + ".json")
    if fault == "marker":
        marker.write_text("f" * 32, "ascii")
    elif fault == "binding":
        binding.write_text("{}", "ascii")
    elif fault == "telemetry":
        (root / (key + ".telemetry-uncertain")).write_text("uncertain", "ascii")
    elif fault == "sidecar":
        Path(str(isolated["old"].path) + "-journal").write_bytes(b"unknown sidecar")
    elif fault == "new_used":
        isolated["new"].prepare(MapSwitchIntent("x", "epoch", "uid", "map", "a" * 64))
    writer = None
    try:
        if fault == "writer":
            writer = sqlite3.connect(isolated["old"].path, isolation_level=None)
            writer.execute("BEGIN IMMEDIATE")
        with pytest.raises((ValueError, OSError, sqlite3.OperationalError)):
            _retire(isolated)
        assert marker.exists()
        assert binding.read_bytes() != map_switch_retirement._SENTINEL
    finally:
        if writer is not None:
            writer.close()


def test_existing_endpoint_owner_prevents_retirement(isolated: dict[str, Any]) -> None:
    lock = isolated["root"] / (isolated["key"] + ".lock")
    with _InstanceLock(lock), pytest.raises(OSError, match=r"."):
        _retire(isolated)
    assert isolated["old"].pending()[1] == "dispatching"


def test_port_reservation_failure_preserves_original_records(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def unavailable() -> Any:
        raise OSError("port reservation failed")
        yield

    monkeypatch.setattr(map_switch_retirement, "_reserve_game_ports", unavailable)
    root, key = isolated["root"], isolated["key"]
    before = (root / (key + ".json")).read_bytes(), (root / (key + ".pending")).read_bytes()
    with pytest.raises(OSError, match="port reservation failed"):
        _retire(isolated)
    assert (
        (root / (key + ".json")).read_bytes(),
        (root / (key + ".pending")).read_bytes(),
    ) == before
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]


def test_directory_handle_identity_mismatch_fails_before_mutation(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(map_switch_retirement, "_directory_handle_identity", lambda _: (-1, -1))
    with pytest.raises(ValueError, match="directory changed while locking"):
        _retire(isolated)
    root, key = isolated["root"], isolated["key"]
    assert (root / (key + ".pending")).exists()
    assert (root / (key + ".json")).read_bytes() != map_switch_retirement._SENTINEL
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]


def test_locked_journal_handle_identity_mismatch_fails_before_mutation(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        map_switch_retirement,
        "_open_read_only_locked",
        lambda _: isolated["attestation"].evidence_path.open("rb"),
    )
    with pytest.raises(ValueError, match="journal changed before locking"):
        _retire(isolated)
    root, key = isolated["root"], isolated["key"]
    assert (root / (key + ".pending")).exists()
    assert (root / (key + ".json")).read_bytes() != map_switch_retirement._SENTINEL


def test_open_evidence_writer_refuses_retirement(isolated: dict[str, Any]) -> None:
    evidence = isolated["attestation"].evidence_path
    with evidence.open("ab"), pytest.raises(OSError, match=r"."):
        _retire(isolated)
    root, key = isolated["root"], isolated["key"]
    assert (root / (key + ".pending")).exists()
    assert (root / (key + ".json")).read_bytes() != map_switch_retirement._SENTINEL


def test_archive_readback_mismatch_fails_before_sentinel(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    write = map_switch_retirement._write_new

    def corrupt(path: Path, data: bytes) -> None:
        write(path, data)
        if path.name == "pending":
            path.write_bytes(b"corrupt")

    monkeypatch.setattr(map_switch_retirement, "_write_new", corrupt)
    with pytest.raises(ValueError, match="readback differs"):
        _retire(isolated)
    root, key = isolated["root"], isolated["key"]
    assert (root / (key + ".pending")).exists()
    assert (root / (key + ".json")).read_bytes() != map_switch_retirement._SENTINEL
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]


def test_final_sync_failure_restores_sentinel(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = map_switch_retirement._sync_registry_directory
    fail_next = False

    def phase(name: str) -> None:
        nonlocal fail_next
        if name == "pending_retired":
            fail_next = True

    def fail_once(path: Path) -> None:
        nonlocal fail_next
        if fail_next:
            fail_next = False
            raise OSError("final sync failed")
        sync(path)

    monkeypatch.setattr(map_switch_retirement, "_phase", phase)
    monkeypatch.setattr(map_switch_retirement, "_sync_registry_directory", fail_once)
    with pytest.raises(OSError, match="final sync failed"):
        _retire(isolated)
    binding = isolated["root"] / (isolated["key"] + ".json")
    assert binding.read_bytes() == map_switch_retirement._SENTINEL
    for journal in (isolated["old"], isolated["new"]):
        with pytest.raises(MapSwitchIndeterminateError):
            MapSwitchController(journal, isolated["endpoint"])


def test_double_publication_failure_is_explicitly_uncertain(
    isolated: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = map_switch_retirement._sync_registry_directory
    replace = map_switch_retirement.os.replace
    after_pending = False
    final_installed = False

    def phase(name: str) -> None:
        nonlocal after_pending
        if name == "pending_retired":
            after_pending = True

    def fail_sync(path: Path) -> None:
        if after_pending:
            raise OSError("final sync failed")
        sync(path)

    def fail_restore(source: Any, destination: Any) -> None:
        nonlocal final_installed
        if after_pending:
            if final_installed:
                raise OSError("sentinel restore failed")
            final_installed = True
        replace(source, destination)

    monkeypatch.setattr(map_switch_retirement, "_phase", phase)
    monkeypatch.setattr(map_switch_retirement, "_sync_registry_directory", fail_sync)
    monkeypatch.setattr(map_switch_retirement.os, "replace", fail_restore)
    with pytest.raises(OSError, match="final sync failed") as caught:
        _retire(isolated)
    assert any("Could not restore retirement sentinel" in note for note in caught.value.__notes__)
    # A live binding is possible when both the sync and rollback fail; this
    # situation is uncertain and must never be reported as fail-closed.
    binding = isolated["root"] / (isolated["key"] + ".json")
    assert binding.read_bytes() != map_switch_retirement._SENTINEL


def test_hardlinked_journal_is_rejected_before_publication(isolated: dict[str, Any]) -> None:
    link = isolated["root"].parent / "hardlink.sqlite"
    os.link(isolated["old"].path, link)
    with pytest.raises(ValueError, match="unsafe or linked"):
        _retire(isolated)
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]


def test_evidence_cannot_alias_the_pending_marker(isolated: dict[str, Any]) -> None:
    marker = isolated["root"] / (isolated["key"] + ".pending")
    isolated["attestation"] = RetirementAttestation(True, True, True, True, marker)
    with pytest.raises(ValueError, match="separate from registry"):
        _retire(isolated)
    assert marker.exists()


def test_symlinked_journal_parent_is_rejected(isolated: dict[str, Any]) -> None:
    parent = isolated["root"].parent
    link = parent / "linked-parent"
    try:
        link.symlink_to(parent, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks requires privileges")
    isolated["new"].path = link / "new.sqlite"
    with pytest.raises((ValueError, OSError, RuntimeError)):
        _retire(isolated)
    assert isolated["old"].path.read_bytes() == isolated["old_bytes"]


def test_reserved_port_does_not_accept_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    # Port 0 obtains an unused ephemeral port; no fixed game or training ports are used.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(map_switch_retirement, "_CONNECT_PORTS", (port,))
    with map_switch_retirement._reserve_game_ports(), socket.socket() as client:
        client.settimeout(1)
        assert client.connect_ex(("127.0.0.1", port)) != 0


def test_rejects_unattested_and_aliased_journal(isolated: dict[str, Any]) -> None:
    isolated["attestation"] = RetirementAttestation(
        False, True, True, True, isolated["attestation"].evidence_path
    )
    with pytest.raises(ValueError, match="attestations"):
        _retire(isolated)
    isolated["attestation"] = RetirementAttestation(
        True, True, True, True, isolated["attestation"].evidence_path
    )
    isolated["new"] = isolated["old"]
    with pytest.raises(ValueError, match="distinct"):
        _retire(isolated)
    assert isolated["old"].pending()[1] == "dispatching"
