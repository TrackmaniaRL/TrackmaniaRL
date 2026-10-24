from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from trackmaniarl.trackmania.native_headless._runtime import ownership, supervisor


def identity(pid: int, *, parent: int = 1, birth: int = 100) -> dict[str, Any]:
    return {
        "pid": pid,
        "ppid": parent,
        "birth": birth,
        "uid": 1000,
        "sid": pid,
        "pgid": pid,
        "state": "S",
    }


def test_pidfd_signal_rechecks_identity_and_closes_fd(monkeypatch: pytest.MonkeyPatch) -> None:
    owned = identity(42)
    calls = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: 123, raising=False)
    monkeypatch.setattr(os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(
        signal, "pidfd_send_signal", lambda fd, sig: calls.append((fd, sig)), raising=False
    )
    monkeypatch.setattr(ownership, "process_identity", lambda pid: identity(pid, birth=200))
    assert ownership.signal_identity(owned, signal.SIGTERM) is False
    assert calls == [("close", 123)]


def test_descendants_keep_session_escape_and_reject_reused_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = {10: identity(10), 20: identity(20, parent=10), 30: identity(30, parent=20)}
    rows[99] = identity(99)
    monkeypatch.setattr(Path, "iterdir", lambda path: [Path(str(pid)) for pid in rows])
    monkeypatch.setattr(ownership, "process_identity", rows.get)
    assert {row["pid"] for row in ownership.owned_descendants(rows[10])} == {20, 30}
    assert ownership.owned_descendants(identity(10, birth=200)) == []


LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd/subreaper integration")


def wait_json(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text())
            assert isinstance(value, dict)
            return value
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
    raise AssertionError(f"missing completed process evidence: {path}")


def assert_gone(owned: dict[str, Any]) -> None:
    current = ownership.process_identity(owned["pid"])
    assert not ownership.same_identity(owned, current) or current["state"] in ("Z", "X")


def driver_script(tmp_path: Path) -> Path:
    script = tmp_path / "driver.py"
    script.write_text(
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True)\n"
        "Path(sys.argv[1]).write_text(json.dumps({'driver': os.getpid(), 'child': child.pid}))\n"
        "print('native-byte-protocol', flush=True)\n"
        "if len(sys.argv) > 2: sys.exit(int(sys.argv[2]))\n"
        "time.sleep(60)\n"
    )
    return script


def launch(tmp_path: Path, *, exit_code: int | None = None) -> supervisor.SupervisedProcess:
    args = [sys.executable, str(driver_script(tmp_path)), str(tmp_path / "tree.json")]
    if exit_code is not None:
        args.append(str(exit_code))
    with (tmp_path / "stderr.bin").open("wb") as stream:
        return supervisor.spawn_supervised(
            args,
            cwd=tmp_path,
            env=dict(os.environ),
            stderr=stream,
            evidence=tmp_path / "supervisor.json",
            timeout=5,
        )


@LINUX
def test_stop_reaps_detached_descendant_and_preserves_native_pid(tmp_path: Path) -> None:
    proc = launch(tmp_path)
    try:
        tree = wait_json(tmp_path / "tree.json")
        child = ownership.process_identity(tree["child"])
        assert proc.pid == tree["driver"]
        assert proc.driver_identity["sid"] == proc.driver_identity["pgid"] == proc.pid
        for pid in (proc.pid, child["pid"]):
            # The driver and its children must never retain either lease end.
            assert not any(
                os.readlink(fd).startswith("socket:")
                for fd in (Path("/proc") / str(pid) / "fd").iterdir()
            )
        assert proc.stdout is not None
        assert proc.stdout.readline() == b"native-byte-protocol\n"
        report = proc.stop()
        assert report["remainingOwned"] == []
        assert report["returncode"] == -signal.SIGKILL
        assert report["reason"] == "actor-stop"
        assert_gone(proc.driver_identity)
        assert_gone(child)
        assert proc.stop() == report
    finally:
        proc.stop()
        proc.close_control()


@LINUX
def test_driver_exit_preserves_status_and_cleans_adopted_children(tmp_path: Path) -> None:
    proc = launch(tmp_path, exit_code=7)
    try:
        tree = wait_json(tmp_path / "tree.json")
        assert proc.wait(timeout=5) == 7
        report = wait_json(tmp_path / "supervisor.json")
        assert report["reason"] == "driver-exited"
        assert report["remainingOwned"] == []
        assert tree["child"] in {row["pid"] for row in report["capturedOwned"]}
        for row in report["capturedOwned"]:
            assert_gone(row)
    finally:
        proc.stop()
        proc.close_control()


@LINUX
def test_transport_reset_after_exit_packet_preserves_received_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = launch(tmp_path, exit_code=7)
    try:
        proc.supervisor.wait(timeout=5)
        original_receive = supervisor._receive
        received = False

        def receive_then_reset(control: socket.socket) -> dict[str, Any] | None:
            nonlocal received
            if received:
                raise ConnectionResetError(104, "injected peer reset after exit packet")
            received = True
            return original_receive(control)

        monkeypatch.setattr(supervisor, "_receive", receive_then_reset)
        report = proc.stop()
        assert received
        assert report["returncode"] == 7
        assert report["remainingOwned"] == []
        assert_gone(proc.driver_identity)
    finally:
        proc.close_control()


@LINUX
def test_startup_eof_with_reset_recovers_durable_cleanup_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trackmaniarl.trackmania.native_headless._runtime.immediate import ImmediateSession

    proc = launch(tmp_path, exit_code=7)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == b"native-byte-protocol\n"
        proc.supervisor.wait(timeout=5)

        def reset(control: socket.socket) -> dict[str, Any] | None:
            raise ConnectionResetError(104, "injected reset before exit packet")

        monkeypatch.setattr(supervisor, "_receive", reset)
        session = ImmediateSession.__new__(ImmediateSession)
        session.proc, session.out, session.tmp = proc, tmp_path, tmp_path / "local"
        session.tmp.mkdir()
        session.closed, session.buffer, session.deadline = False, b"", time.monotonic() + 5
        session.captures, session.commands, session.messages = [], [], []
        session.connection = session.listener = session.stderr = None
        with pytest.raises(EOFError, match="driver EOF return=7"):
            session.line()
        session.close(error=True)
        report = json.loads((tmp_path / "close-v7.json").read_bytes())
        assert report["error"] is True
        assert report["failures"] == []
        assert report["supervisor"]["returncode"] == 7
        assert report["supervisor"]["remainingOwned"] == []
        for row in report["supervisor"]["capturedOwned"]:
            assert_gone(row)
    finally:
        proc.close_control()


@LINUX
@pytest.mark.parametrize("fault", ["missing", "driverIdentity", "supervisorIdentity", "symlink"])
def test_transport_reset_cannot_accept_missing_foreign_or_aliased_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    proc = launch(tmp_path, exit_code=7)
    try:
        proc.supervisor.wait(timeout=5)
        evidence = tmp_path / "supervisor.json"
        report = json.loads(evidence.read_bytes())
        if fault == "missing":
            evidence.unlink()
        elif fault == "symlink":
            foreign = tmp_path / "foreign-receipt.json"
            evidence.rename(foreign)
            evidence.symlink_to(foreign)
        else:
            report[fault]["birth"] += 1
            evidence.write_text(json.dumps(report))

        def reset(control: socket.socket) -> dict[str, Any] | None:
            raise ConnectionResetError(104, "injected reset before exit packet")

        monkeypatch.setattr(supervisor, "_receive", reset)
        with pytest.raises(RuntimeError, match="without matching cleanup evidence"):
            proc.stop()
        assert proc.report is None
        for row in report["capturedOwned"]:
            assert_gone(row)
    finally:
        proc.close_control()


@LINUX
def test_fifo_receipt_without_writer_is_rejected_without_blocking(tmp_path: Path) -> None:
    """An isolated alarm bounds the regression even if receipt opening blocks."""
    evidence = tmp_path / "receipt.fifo"
    os.mkfifo(evidence)
    wrapper = tmp_path / "fifo-recovery.py"
    wrapper.write_text(
        "import signal, sys\n"
        "from pathlib import Path\n"
        "from trackmaniarl.trackmania.native_headless._runtime.supervisor "
        "import SupervisedProcess\n"
        "proc = SupervisedProcess.__new__(SupervisedProcess)\n"
        "proc.report = None\n"
        "proc.evidence = Path(sys.argv[1])\n"
        "signal.alarm(2)\n"
        "try:\n"
        "    proc._recover_report()\n"
        "except RuntimeError as error:\n"
        "    assert str(error) == 'native supervisor exited without matching cleanup evidence'\n"
        "    assert isinstance(error.__cause__, ValueError)\n"
        "    assert str(error.__cause__) == 'invalid native supervisor receipt file'\n"
        "else:\n"
        "    raise AssertionError('FIFO was accepted as a cleanup receipt')\n"
        "finally:\n"
        "    signal.alarm(0)\n"
    )
    completed = subprocess.run(
        [sys.executable, str(wrapper), str(evidence)],
        check=False,
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert evidence.is_fifo()


@LINUX
@pytest.mark.parametrize("phase", ["popen-return", "native-started"])
def test_interrupted_factory_closes_lease_and_cleans_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    original_popen = subprocess.Popen
    original_init = supervisor.SupervisedProcess.__init__
    launched = []

    def interrupt_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc = original_popen(*args, **kwargs)
        launched.append(proc)
        if phase == "popen-return":
            raise KeyboardInterrupt
        return proc

    def interrupt_started(self: Any, *args: Any) -> None:
        original_init(self, *args)
        if phase == "native-started":
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", interrupt_popen)
    monkeypatch.setattr(supervisor.SupervisedProcess, "__init__", interrupt_started)
    with pytest.raises(KeyboardInterrupt):
        launch(tmp_path)
    assert launched[0].wait(timeout=5) == 0
    for stream in (launched[0].stdin, launched[0].stdout):
        if stream is not None:
            stream.close()
    if phase == "native-started":
        report = wait_json(tmp_path / "supervisor.json")
        assert report["reason"] == "actor-lease-eof"
        assert report["remainingOwned"] == []
        for row in report["capturedOwned"]:
            assert_gone(row)
    else:
        assert not (tmp_path / "tree.json").exists()


@LINUX
def test_actor_sigkill_cleans_tree_without_signalling_unrelated_process(tmp_path: Path) -> None:
    script = driver_script(tmp_path)
    actor_script = tmp_path / "actor.py"
    actor_script.write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "from trackmaniarl.trackmania.native_headless._runtime.supervisor import spawn_supervised\n"
        "root = Path(sys.argv[1])\n"
        "with (root / 'stderr.bin').open('wb') as stream:\n"
        "    proc = spawn_supervised([sys.executable, sys.argv[2], str(root / 'tree.json')], "
        "cwd=root, env=dict(os.environ), stderr=stream, "
        "evidence=root / 'supervisor.json', timeout=5)\n"
        "    assert proc.stdout.readline() == b'native-byte-protocol\\n'\n"
        "    (root / 'actor-ready.json').write_text(json.dumps(proc.driver_identity))\n"
        "    time.sleep(60)\n"
    )
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    with (tmp_path / "actor.stderr").open("wb") as stream:
        actor = subprocess.Popen(
            [sys.executable, str(actor_script), str(tmp_path), str(script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stream,
        )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        driver = wait_json(tmp_path / "actor-ready.json")
        tree = wait_json(tmp_path / "tree.json")
        child = ownership.process_identity(tree["child"])
        actor.kill()
        actor.wait(timeout=5)
        report = wait_json(tmp_path / "supervisor.json")
        assert report["reason"] == "actor-lease-eof"
        assert report["remainingOwned"] == []
        assert_gone(driver)
        assert_gone(child)
        assert unrelated.poll() is None
        assert unrelated.pid not in {row["pid"] for row in report["cleanupPidfdSignals"]}
    finally:
        if actor.poll() is None:
            actor.kill()
            actor.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@LINUX
def test_missing_pidfd_support_fails_before_any_native_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(pid: int) -> int:
        raise OSError("pidfd unsupported")

    def no_launch(*args: Any, **kwargs: Any) -> None:
        pytest.fail("native launch occurred before pidfd preflight")

    monkeypatch.setattr(os, "pidfd_open", unavailable)
    monkeypatch.setattr(subprocess, "Popen", no_launch)
    control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        with pytest.raises(OSError, match="pidfd unsupported"):
            supervisor._run(control, tmp_path / "unused.json")
    finally:
        control.close()
        child.close()


@LINUX
@pytest.mark.parametrize("fault", ["pidfd", "identity", "publication"])
def test_post_launch_setup_or_reporting_failure_always_cleans_native_tree(
    tmp_path: Path, fault: str
) -> None:
    """Inject faults inside an isolated supervisor, never into the test runner."""
    script = driver_script(tmp_path)
    wrapper = tmp_path / "fault-supervisor.py"
    wrapper.write_text(
        "import os, socket, sys, time\n"
        "from pathlib import Path\n"
        "from trackmaniarl.trackmania.native_headless._runtime import supervisor as s\n"
        "kind, tree = sys.argv[3], Path(sys.argv[4])\n"
        "original_popen = s.subprocess.Popen\n"
        "original_identity, original_pidfd = s.ownership.process_identity, os.pidfd_open\n"
        "target, failed = None, False\n"
        "def launch(*args, **kwargs):\n"
        "    global target\n"
        "    child = original_popen(*args, **kwargs)\n"
        "    target = child.pid\n"
        "    deadline = time.monotonic() + 3\n"
        "    while not tree.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
        "    return child\n"
        "def identify(pid):\n"
        "    global failed\n"
        "    if kind == 'identity' and pid == target and not failed:\n"
        "        failed = True\n"
        "        raise OSError('injected identity capture failure')\n"
        "    return original_identity(pid)\n"
        "def pidfd(pid, *args):\n"
        "    global failed\n"
        "    if kind == 'pidfd' and pid == target and not failed:\n"
        "        failed = True\n"
        "        raise OSError('injected pidfd allocation failure')\n"
        "    return original_pidfd(pid, *args)\n"
        "s.subprocess.Popen = launch\n"
        "s.ownership.process_identity = identify\n"
        "os.pidfd_open = pidfd\n"
        "with socket.socket(fileno=int(sys.argv[1])) as lease:\n"
        "    raise SystemExit(s._run(lease, Path(sys.argv[2])))\n"
    )
    evidence = tmp_path / "supervisor.json"
    if fault == "publication":
        evidence.write_text("preserved-original")
    control, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    control.settimeout(15)
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    with (tmp_path / "supervisor.stderr").open("wb") as errors:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(wrapper),
                str(child.fileno()),
                str(evidence),
                fault,
                str(tmp_path / "tree.json"),
            ],
            env=env,
            pass_fds=(child.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            start_new_session=True,
        )
    child.close()
    try:
        control.sendall(
            json.dumps(
                {
                    "kind": "start",
                    "args": [sys.executable, str(script), str(tmp_path / "tree.json")],
                    "cwd": str(tmp_path),
                    "env": env,
                }
            ).encode()
        )
        report = json.loads(control.recv(1024 * 1024))
        if fault == "publication":
            assert report["kind"] == "started"
            control.sendall(b'{"kind":"stop"}')
            report = json.loads(control.recv(1024 * 1024))
            assert report["publicationError"]
            assert evidence.read_text() == "preserved-original"
        else:
            assert "injected" in report["supervisorError"]
            assert report["reason"] == "supervisor-error"
        assert report["kind"] == "exited"
        assert report["remainingOwned"] == []
        assert report["cleanupErrors"] == []
        assert proc.wait(timeout=5) == 1
        tree = wait_json(tmp_path / "tree.json")
        captured = {row["pid"]: row for row in report["capturedOwned"]}
        for pid in tree.values():
            assert pid in captured
            assert_gone(captured[pid])
    finally:
        control.close()
        proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                stream.close()
