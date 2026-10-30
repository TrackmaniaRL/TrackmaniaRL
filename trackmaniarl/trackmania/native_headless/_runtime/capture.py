from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from trackmaniarl.trackmania.native_headless._runtime.shapes import CaptureSpec


def sha(data: Any) -> Any:
    return hashlib.sha256(data).hexdigest()


def json_exclusive(path: Any, value: Any) -> Any:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class RxPersistenceError(OSError):
    secondary_errors: list[Exception]
    evidence: Any
    "Fatal storage error, deliberately NOT an IncompleteReceive/EOFError."

    def __init__(self, message: Any, evidence: Any = None) -> None:
        super().__init__(message)
        self.evidence = evidence


class Capture:
    chunks: Any
    closed: Any
    ep: Any
    exact_calls: Any
    fd: Any
    local: Any
    persisted: Any
    pid: Any
    provenance: Any
    publish_error: Any
    published: Any
    received: Any
    stage: Any
    write_calls: Any

    def __init__(self, spec: CaptureSpec) -> None:
        ep, pid, tmp, provenance = (spec.ep, spec.pid, spec.tmp, spec.provenance)
        self.ep = Path(ep)
        self.pid = pid
        self.provenance = deepcopy(provenance)
        root = Path(tmp).resolve()
        if not root.is_relative_to(Path("/tmp")):
            raise RxPersistenceError("RX009 requires the owned Linux /tmp session directory")
        directory = root / "rx009"
        directory.mkdir(exist_ok=True)
        self.local = directory / f"{self.ep.name}-pid{pid}.bin"
        self.fd = self.local.open("xb", buffering=0)
        self.received = 0
        self.persisted = 0
        self.chunks = 0
        self.write_calls = 0
        self.exact_calls = 0
        self.closed = False
        self.published = None
        self.publish_error = None
        self.stage = None

    def location(self) -> Any:
        json_exclusive(
            self.ep / "rx009-location.json",
            {
                "schema": "tmre.rx009-location/v1",
                "pid": self.pid,
                "localRxFile": str(self.local),
                "destinationRxFile": str(self.ep / "rx.bin"),
                "unbuffered": True,
                "beforeFirstDecode": True,
                "activeCaller009": self.provenance,
                "durabilityFsync": False,
                "retainedLocalEvidence": True,
            },
        )

    def append(self, chunk: Any) -> Any:
        self.received += len(chunk)
        self.chunks += 1
        view = memoryview(chunk)
        while view:
            written = self.fd.write(view)
            self.write_calls += 1
            if type(written) is not int or written <= 0 or written > len(view):
                raise RxPersistenceError("unbuffered RX write made no valid progress")
            self.persisted += written
            view = view[written:]

    def close_fd(self) -> Any:
        if not self.closed:
            try:
                self.fd.close()
            except BaseException:
                self.closed = bool(getattr(self.fd, "closed", False))
                raise
            else:
                self.closed = True

    def finish(self) -> Any:
        if self.publish_error is not None:
            self.close_fd()
            raise self.publish_error
        if self.published is not None:
            raw = (self.ep / "rx.bin").read_bytes()
            if len(raw) != self.published["bytes"] or sha(raw) != self.published["sha256"]:
                raise RxPersistenceError("already published RX no longer matches its receipt")
            return self.published
        try:
            self.close_fd()
            self.stage = self.ep / ("rx009-stage-" + uuid.uuid4().hex + ".bin")
            digest = hashlib.sha256()
            size = 0
            with self.local.open("rb") as source, self.stage.open("xb", buffering=0) as dest:
                while True:
                    data = source.read(65536)
                    if not data:
                        break
                    digest.update(data)
                    size += len(data)
                    view = memoryview(data)
                    while view:
                        n = dest.write(view)
                        if type(n) is not int or n <= 0 or n > len(view):
                            raise RxPersistenceError("RX copy made no valid progress")
                        view = view[n:]
            copied = self.stage.read_bytes()
            if size != self.persisted or len(copied) != size or sha(copied) != digest.hexdigest():
                raise RxPersistenceError("RX copy size/hash mismatch")
            os.link(self.stage, self.ep / "rx.bin")
            final = (self.ep / "rx.bin").read_bytes()
            if len(final) != size or sha(final) != digest.hexdigest():
                raise RxPersistenceError("published RX size/hash mismatch")
            receipt = {
                "schema": "tmre.rx009-publication/v1",
                "pid": self.pid,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "localRxFile": str(self.local),
                "stageFile": str(self.stage),
                "destinationRxFile": str(self.ep / "rx.bin"),
                "receivedBytes": self.received,
                "persistedBytes": self.persisted,
                "nonemptyRecvChunks": self.chunks,
                "writeCalls": self.write_calls,
                "exactCalls": self.exact_calls,
                "unbuffered": True,
                "durabilityFsync": False,
                "allReceivedBytesPersisted": self.received == self.persisted,
                "activeCaller009": self.provenance,
                "noClobberPublish": "same-directory os.link; no overwrite fallback",
                "retainedLocalEvidence": True,
            }
            json_exclusive(self.ep / "rx009-publication.json", receipt)
            self.published = receipt
            return receipt
        except Exception as error:
            self.publish_error = RxPersistenceError("RX publication failed: " + repr(error))
            raise self.publish_error from error
