"""Opt-in protocol-3 switch controller for experimental distributed evaluation.

A terminal UID/readiness acknowledgement is not proof of loaded map bytes.
Use only with an exclusively owned, separately qualified game instance.
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import json
import os
import socket
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

from trackmaniarl.trackmania.map_switch_journal import MapSwitchIntent, MapSwitchJournal


class MapSwitchIndeterminateError(RuntimeError):
    """The game may still execute a map load; do not resume collection."""


# Connect gives both the return-to-menu and map-load phases up to 30 seconds.
# Allow both phases to complete, plus time for status requests and scheduling.
_DEFAULT_SWITCH_TIMEOUT_S = 75.0
_MAX_SWITCH_TIMEOUT_S = 90.0


def _registry_root() -> Path:
    """One fixed per-user namespace for all configured game instances."""
    return Path.home() / ".trackmaniarl-map-switch-registry-v1"


def _sync_registry_directory(root: Path) -> None:
    """Sync directory entries on POSIX; Windows durability is not established."""
    if os.name == "nt":
        return
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _InstanceLock:
    """Hold a local OS lock for the full lifetime of an actor's game session."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("a+b")
        try:
            opened = os.fstat(self.file.fileno())
            entry = path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or stat.S_ISLNK(entry.st_mode)
                or getattr(entry, "st_file_attributes", 0) & 0x400
                or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
            ):
                raise OSError("unsafe game instance lock file")
            self.file.seek(0)
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise

    def close(self) -> None:
        if self.file.closed:
            return
        try:
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()

    def __enter__(self) -> _InstanceLock:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _unique_fields(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate switch response field")
        result[key] = value
    return result


class OpenPlanetSwitchTransport:
    """Bounded one-request JSONL transport to an explicitly owned local instance."""

    def __init__(self, host: str, port: int, timeout_s: float = 3.0) -> None:
        address = ipaddress.ip_address(host)
        if not address.is_loopback or not 1 <= port <= 65535:
            raise ValueError("switch endpoint must be an explicit loopback address and port")
        if not 0 < timeout_s <= 10:
            raise ValueError("switch transport timeout must be in (0, 10]")
        self.host, self.port, self.timeout_s = str(address), port, timeout_s

    def __call__(self, request: Mapping[str, str]) -> dict[str, Any]:
        encoded = json.dumps(request, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > 16384:
            raise ValueError("switch request exceeds protocol limit")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            sock.sendall(encoded)
            data = bytearray()
            while not data.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("switch response disconnected")
                data.extend(chunk)
                if len(data) > 16384:
                    raise ValueError("switch response exceeds protocol limit")
        decoded = json.loads(data, object_pairs_hook=_unique_fields)
        if not isinstance(decoded, dict):
            raise ValueError("invalid switch response")
        return decoded


class MapSwitchController:
    """Durable send-before-effect protocol with explicit recovery of ambiguous calls.

    A controller does not clear unresolved records or infer success on timeout.
    Every client for a game instance must share the same journal and lease.
    """

    def __init__(
        self, journal: MapSwitchJournal, exchange: Callable[[Mapping[str, str]], Mapping[str, Any]]
    ) -> None:
        self.journal, self.exchange = journal, exchange
        root = _registry_root()
        root.mkdir(parents=True, exist_ok=True)
        identity = (
            f"endpoint:{exchange.host}:{exchange.port}"
            if isinstance(exchange, OpenPlanetSwitchTransport)
            else f"instance:{journal.instance_id}"
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self._quarantine_path = root / (key + ".pending")
        try:
            self._instance_lock = _InstanceLock(root / (key + ".lock"))
        except OSError as error:
            raise MapSwitchIndeterminateError("game instance is already owned") from error
        try:
            self._bind_journal(root / (key + ".json"))
        except BaseException:
            self._instance_lock.close()
            raise

    def _bind_journal(self, record: Path) -> None:
        """Never allow a second journal for an already bound instance ID."""
        expected = json.dumps(
            {
                "instance_id": self.journal.instance_id,
                "journal": os.path.normcase(str(self.journal.path)),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            descriptor = os.open(record, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as stream:
                if stream.write(expected) != len(expected):
                    raise OSError("short write while binding switch journal")
                stream.flush()
                os.fsync(stream.fileno())
        entry = record.lstat()
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or getattr(entry, "st_file_attributes", 0) & 0x400
        ):
            raise MapSwitchIndeterminateError("unsafe game instance registry record")
        with record.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
                raise MapSwitchIndeterminateError("game instance registry record changed")
            actual = stream.read(4097)
        if actual != expected:
            raise MapSwitchIndeterminateError("game instance is bound to a different journal")

    def close(self) -> None:
        """Release ownership only when the actor has stopped using the game."""
        self._instance_lock.close()

    def __enter__(self) -> MapSwitchController:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _require_owner(self) -> None:
        if self._instance_lock.file.closed:
            raise MapSwitchIndeterminateError("game instance ownership was released")

    def _read_quarantine(self) -> str | None:
        try:
            entry = self._quarantine_path.lstat()
        except FileNotFoundError:
            return None
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or getattr(entry, "st_file_attributes", 0) & 0x400
        ):
            raise MapSwitchIndeterminateError("unsafe switch quarantine marker")
        with self._quarantine_path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
                raise MapSwitchIndeterminateError("switch quarantine marker changed")
            raw = stream.read(33)
        try:
            operation = raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise MapSwitchIndeterminateError("invalid switch quarantine marker") from error
        if len(operation) != 32 or any(char not in "0123456789abcdef" for char in operation):
            raise MapSwitchIndeterminateError("invalid switch quarantine marker")
        return operation

    def _mark_quarantine(self, operation_id: str) -> None:
        try:
            descriptor = os.open(self._quarantine_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise MapSwitchIndeterminateError("a previous switch is quarantined") from error
        with os.fdopen(descriptor, "wb") as stream:
            data = operation_id.encode("ascii")
            if stream.write(data) != len(data):
                raise OSError("short write to switch quarantine marker")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_registry_directory(self._quarantine_path.parent)

    def _clear_quarantine(self, operation_id: str) -> None:
        if self._read_quarantine() != operation_id:
            raise MapSwitchIndeterminateError("switch quarantine marker differs from operation")
        self._quarantine_path.unlink()
        _sync_registry_directory(self._quarantine_path.parent)

    def _send(self, command: str, **fields: str) -> Mapping[str, Any]:
        response = self.exchange({"protocol_version": "3", "command": command, **fields})
        if response.get("status") != "ok" or response.get("protocol_version") != "3":
            raise MapSwitchIndeterminateError("switch endpoint rejected or malformed request")
        return response

    def _instance(self) -> str:
        response = self._send("get_instance")
        value = response.get("instance_epoch")
        if not isinstance(value, str) or not value or len(value) > 128:
            raise MapSwitchIndeterminateError("plugin instance epoch is unavailable")
        return value

    @staticmethod
    def _epochs(intent: MapSwitchIntent) -> tuple[str, str]:
        try:
            value = json.loads(intent.plugin_epoch)
        except (ValueError, TypeError) as error:
            raise MapSwitchIndeterminateError("journal epoch binding is invalid") from error
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise MapSwitchIndeterminateError("journal epoch binding is invalid")
        return value[0], value[1]

    def switch(  # noqa: PLR0913 -- timeout and optional quiescing are independent
        self,
        map_path: Path,
        uid: str,
        *,
        timeout_s: float = _DEFAULT_SWITCH_TIMEOUT_S,
        before_dispatch: Callable[[], None] | None = None,
    ) -> None:
        """Issue one switch; any uncertain outcome remains durably quarantined."""
        self._require_owner()
        self._validate_timeout(timeout_s)
        if self._read_quarantine() is not None or self.journal.pending() is not None:
            raise MapSwitchIndeterminateError("a previous switch is unresolved")
        path = map_path.resolve(strict=True)
        if not path.is_file() or not uid.strip():
            raise ValueError("an existing canonical map and UID are required")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        instance = self._instance()
        latest = self.journal.latest()
        if latest is None:
            lease = uuid4().hex
        else:
            previous_instance, lease = self._epochs(latest[0])
            if previous_instance != instance:
                raise MapSwitchIndeterminateError("plugin restarted; instance needs recovery")
        intent = MapSwitchIntent(uuid4().hex, json.dumps([instance, lease]), uid, str(path), digest)
        # The independent marker must exist before any database mutation or
        # game request. A rolled-back or replaced database cannot erase it.
        self._mark_quarantine(intent.operation_id)
        self.journal.prepare(intent)
        try:
            response = self._send("claim", instance_epoch=instance, epoch=lease)
            self._verify_identity(response, instance, lease)
            if before_dispatch is not None:
                before_dispatch()
            self.journal.begin_dispatch(intent.operation_id)
            response = self._send(
                "switch_map",
                instance_epoch=instance,
                epoch=lease,
                operation_id=intent.operation_id,
                map_path=intent.target_path,
                expected_map_uid=uid,
            )
            self._wait(intent, response, timeout_s)
        except Exception as error:
            if isinstance(error, MapSwitchIndeterminateError):
                raise
            raise MapSwitchIndeterminateError(
                "switch outcome unknown; session quarantined"
            ) from error

    def recover(self, *, timeout_s: float = _DEFAULT_SWITCH_TIMEOUT_S) -> None:
        """Poll the same dispatched operation; never issue another switch."""
        self._require_owner()
        self._validate_timeout(timeout_s)
        marker = self._read_quarantine()
        pending = self.journal.pending()
        if pending is None:
            if marker is not None:
                latest = self.journal.latest()
                if latest is None or latest[1] != "terminal" or latest[0].operation_id != marker:
                    raise MapSwitchIndeterminateError("journal and quarantine marker disagree")
                self._recover_terminal_marker(latest[0])
            return
        intent, state = pending
        if marker != intent.operation_id:
            raise MapSwitchIndeterminateError("journal and quarantine marker disagree")
        if state != "dispatching":
            raise MapSwitchIndeterminateError("prepared operation requires explicit recovery")
        instance, lease = self._epochs(intent)
        try:
            if self._instance() != instance:
                raise MapSwitchIndeterminateError(
                    "plugin epoch changed while switch was unresolved"
                )
            response = self._send(
                "switch_status",
                instance_epoch=instance,
                epoch=lease,
                operation_id=intent.operation_id,
            )
            self._wait(intent, response, timeout_s)
        except Exception as error:
            if isinstance(error, MapSwitchIndeterminateError):
                raise
            raise MapSwitchIndeterminateError(
                "switch recovery failed; session quarantined"
            ) from error

    def _recover_terminal_marker(self, intent: MapSwitchIntent) -> None:
        """Recheck a terminal operation after a crash before marker removal."""
        instance, lease = self._epochs(intent)
        try:
            if self._instance() != instance:
                raise MapSwitchIndeterminateError("plugin epoch changed before marker cleanup")
            with Path(intent.target_path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != intent.target_sha256:
                raise MapSwitchIndeterminateError("map file changed before marker cleanup")
            response = self._send(
                "switch_status",
                instance_epoch=instance,
                epoch=lease,
                operation_id=intent.operation_id,
            )
            self._require_ready(response, intent, instance, lease)
            self._clear_quarantine(intent.operation_id)
        except MapSwitchIndeterminateError:
            raise
        except Exception as error:
            raise MapSwitchIndeterminateError("terminal marker recovery failed") from error

    def verify_active_map(self, expected_uid: str, *, require_player: bool = True) -> None:
        """Reconfirm the terminal training map immediately before collection.

        The caller must retain exclusive control between this check and use.
        """
        self._active_map_status(expected_uid, require_player=require_player)

    def require_capacity(self, expected_uid: str, required: int) -> None:
        """Check there is room for a whole suite and its training-map restore.

        The caller must retain exclusive ownership through all reserved switches.
        Missing or malformed capacity is rejected before dispatching a switch.
        """
        if type(required) is not int or not 1 <= required <= 4096:
            raise ValueError("required switch capacity must be an integer in [1, 4096]")
        response = self._active_map_status(expected_uid, require_player=False)
        remaining = response.get("remaining_operations")
        if type(remaining) is not int or not 0 <= remaining <= 4096:
            raise MapSwitchIndeterminateError("switch endpoint did not report valid capacity")
        if remaining < required:
            raise MapSwitchIndeterminateError("insufficient switch capacity for suite and restore")

    def _active_map_status(self, expected_uid: str, *, require_player: bool) -> Mapping[str, Any]:
        self._require_owner()
        if self._read_quarantine() is not None or self.journal.pending() is not None:
            raise MapSwitchIndeterminateError("a previous switch is unresolved")
        latest = self.journal.latest()
        if latest is None or latest[1] != "terminal" or latest[0].target_uid != expected_uid:
            raise MapSwitchIndeterminateError("training map lacks a matching terminal switch")
        intent = latest[0]
        instance, lease = self._epochs(intent)
        try:
            if self._instance() != instance:
                raise MapSwitchIndeterminateError("plugin restarted before collection")
            with Path(intent.target_path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != intent.target_sha256:
                raise MapSwitchIndeterminateError("training map file changed")
            response = self._send(
                "switch_status",
                instance_epoch=instance,
                epoch=lease,
                operation_id=intent.operation_id,
            )
            self._require_ready(response, intent, instance, lease, require_player=require_player)
            return response
        except Exception as error:
            if isinstance(error, MapSwitchIndeterminateError):
                raise
            raise MapSwitchIndeterminateError("training map verification failed") from error

    @staticmethod
    def _verify_identity(response: Mapping[str, Any], instance: str, lease: str) -> None:
        if response.get("instance_epoch") != instance or response.get("epoch") != lease:
            raise MapSwitchIndeterminateError("switch response epoch differs")

    @staticmethod
    def _validate_timeout(timeout_s: float) -> None:
        if not 0 < timeout_s <= _MAX_SWITCH_TIMEOUT_S:
            raise ValueError("switch timeout must be in (0, 90]")

    def _require_ready(  # noqa: PLR0913 -- response and independently bound epochs
        self,
        response: Mapping[str, Any],
        intent: MapSwitchIntent,
        instance: str,
        lease: str,
        *,
        require_player: bool = True,
    ) -> None:
        self._verify_identity(response, instance, lease)
        if (
            response.get("operation_id") != intent.operation_id
            or response.get("state") != "ready"
            or response.get("map_uid") != intent.target_uid
            or (require_player and response.get("ready") != "true")
        ):
            raise MapSwitchIndeterminateError(
                "ready response lacks matching operation, map and player"
            )

    def _wait(self, intent: MapSwitchIntent, response: Mapping[str, Any], timeout_s: float) -> None:
        instance, lease = self._epochs(intent)
        deadline = monotonic() + timeout_s
        while True:
            self._verify_identity(response, instance, lease)
            if response.get("operation_id") != intent.operation_id:
                raise MapSwitchIndeterminateError("switch operation identity differs")
            state = response.get("state")
            if state == "ready":
                self._require_ready(response, intent, instance, lease)
                with Path(intent.target_path).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != intent.target_sha256:
                    raise MapSwitchIndeterminateError("map file changed during switch")
                response = self._send(
                    "switch_status",
                    instance_epoch=instance,
                    epoch=lease,
                    operation_id=intent.operation_id,
                )
                self._require_ready(response, intent, instance, lease)
                self.journal.resolve_terminal(intent.operation_id, intent.plugin_epoch, "ready")
                self._clear_quarantine(intent.operation_id)
                return
            if state not in ("accepted", "leaving", "loading") or monotonic() >= deadline:
                raise MapSwitchIndeterminateError("switch did not reach a verified terminal state")
            sleep(min(0.1, max(0.0, deadline - monotonic())))
            response = self._send(
                "switch_status",
                instance_epoch=instance,
                epoch=lease,
                operation_id=intent.operation_id,
            )
