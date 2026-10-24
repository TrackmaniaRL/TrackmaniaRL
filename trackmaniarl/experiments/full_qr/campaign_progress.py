"""Independent real-counter heartbeat, separate from sparse training metric logs."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
from pathlib import Path
from time import monotonic
from typing import Any


class WorkerProgress:
    def __init__(  # noqa: PLR0913 - independent coordinator, destination, stop and diagnostic cadence.
        self, coordinator: Any, path: Path, stop: Any, interval: float = 1.0
    ) -> None:
        self.coordinator, self.path, self.stop = coordinator, path, stop
        self.interval = interval
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="qr-worker-progress", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.done.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("worker counter reporter did not close")

    def _watch(self) -> None:
        previous, changed, dumped = -1, monotonic(), False
        try:
            while not self.done.wait(self.interval):
                counters = self.coordinator.counters
                now = monotonic()
                if counters.updates != previous:
                    previous, changed, dumped = counters.updates, now, False
                elif counters.updates > 0 and now - changed > 10 and not dumped:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    dumped = True
                row = {
                    "pid": os.getpid(),
                    "monotonic": now,
                    "transitions": counters.transitions,
                    "updates": counters.updates,
                    "episodes": counters.episodes,
                    "update_credit": counters.update_credit,
                }
                temporary = self.path.with_suffix(".stage")
                temporary.write_text(json.dumps(row, allow_nan=False))
                for attempt in range(10):
                    try:
                        os.replace(temporary, self.path)
                        break
                    except PermissionError:
                        # Windows readers may briefly deny replacement; ext4
                        # does not. Never turn an unbounded I/O failure into a
                        # silently stale heartbeat.
                        if attempt == 9:
                            raise
                        if self.done.wait(0.01):
                            temporary.unlink(missing_ok=True)
                            return
        except BaseException:
            self.stop.set()
            raise
