"""Read static Openplanet API dumps and installed source archives without loading them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
    }


def inventory(root: Path) -> dict[str, Any]:
    core = json.loads((root / "OpenplanetCore.json").read_bytes())
    game = json.loads((root / "OpenplanetNext.json").read_bytes())
    relevant = re.compile(r"Input|Simula|Tick|RaceTime|GameTime|TimeSince|PlayerState|SceneVehicle")
    classes = {}
    for namespace, rows in game["ns"].items():
        for name, row in rows.items():
            if relevant.search(name):
                classes[f"{namespace}::{name}"] = row
    plugins = []
    hook_pattern = re.compile(
        r"Dev::(?:Hook|Unhook|InterceptProc|ResetInterceptProc|FindPattern)|HookInfo|ProcIntercept"
    )
    for path in sorted((root / "Plugins").glob("*.op")):
        row: dict[str, Any] = {**identity(path), "hook_examples": []}
        with zipfile.ZipFile(path) as archive:
            row["files"] = [
                {"name": item.filename, "bytes": item.file_size} for item in archive.infolist()
            ]
            for name in archive.namelist():
                if name == "info.toml":
                    row["manifest"] = archive.read(name).decode("utf-8-sig")
                if name.endswith((".as", ".md", ".toml")):
                    lines = archive.read(name).decode("utf-8-sig").splitlines()
                    for index, text in enumerate(lines):
                        if hook_pattern.search(text):
                            row["hook_examples"].append(
                                {
                                    "file": name,
                                    "line": index + 1,
                                    "context": lines[max(0, index - 3) : index + 6],
                                }
                            )
        plugins.append(row)
    return {
        "schema": "tmrl.windows-static-input-hook-inventory/v1",
        "live_process_accessed": False,
        "game_controlled": False,
        "api_snapshots": [
            identity(root / name)
            for name in ("Openplanet.h", "OpenplanetCore.json", "OpenplanetNext.json")
        ],
        "core_version": core["op"],
        "game_version": {"op": game["op"], "mp": game["mp"]},
        "dev_hook_functions": [
            item
            for item in core["functions"]
            if item.get("ns") == "Dev"
            and re.search(r"Hook|Intercept|FindPattern|BaseAddress", item["name"])
        ],
        "dev_callbacks": [item for item in core["funcdefs"] if item.get("ns") == "Dev"],
        "core_stack_and_hook_types": [
            item for item in core["classes"] if item["name"] in ("HookInfo", "CMwStack")
        ],
        "dev_hook_enums": [item for item in core["enums"] if item.get("ns") == "Dev"],
        "input_and_simulation_classes": classes,
        "plugins": plugins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.snapshot_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "core_version": report["core_version"],
                "plugins": [
                    {
                        "path": row["path"],
                        "manifest": row.get("manifest"),
                        "hook_examples": len(row["hook_examples"]),
                    }
                    for row in report["plugins"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
