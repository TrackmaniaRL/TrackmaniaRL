from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from scripts.benchmark_replay_roadmap import create_output_directory


def test_concurrent_benchmarks_cannot_claim_same_output(tmp_path: Path) -> None:
    target = tmp_path / "fresh" / "run"
    start = Barrier(2)

    def claim() -> str:
        start.wait(timeout=5)
        try:
            create_output_directory(target)
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))
    assert sorted(results) == ["created", "exists"]
    assert target.is_dir()
