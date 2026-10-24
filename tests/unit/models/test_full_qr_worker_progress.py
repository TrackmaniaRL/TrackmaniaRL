import json
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

from trackmaniarl.experiments.full_qr.campaign_progress import WorkerProgress


def test_real_counter_progress_does_not_depend_on_metric_log_cadence(tmp_path: Path) -> None:
    counters = SimpleNamespace(transitions=10000, updates=0, episodes=3, update_credit=0.0)
    signal = Event()
    path = tmp_path / "progress.json"
    progress = WorkerProgress(SimpleNamespace(counters=counters), path, signal, interval=0.01)
    progress.start()
    try:
        deadline = monotonic() + 3
        while not path.exists() and monotonic() < deadline:
            sleep(0.01)
        assert json.loads(path.read_text())["updates"] == 0
        counters.updates = 7
        while json.loads(path.read_text())["updates"] != 7 and monotonic() < deadline:
            sleep(0.01)
        assert json.loads(path.read_text())["updates"] == 7
        assert not signal.is_set()
    finally:
        progress.close()
    assert not progress.thread.is_alive()
    assert not path.with_suffix(".stage").exists()
