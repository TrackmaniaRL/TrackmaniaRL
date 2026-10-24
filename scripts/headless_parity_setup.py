"""Make a new ext4 development deployment; never modify the qualified baseline."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.diagnostic_inputs import configured_path


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    baseline = Path(str(configured_path("TRACKMANIARL_NATIVE_BASELINE")))
    target = Path(sys.argv[1])
    if sys.platform != "linux" or not target.is_absolute() or str(target).startswith("/mnt/"):
        raise ValueError("a fresh absolute Linux/ext4 target is required")
    if target.exists() or target.resolve() != target:
        raise ValueError("target exists or is aliased")
    target.mkdir(mode=0o700)
    shutil.copytree(
        source / "trackmaniarl",
        target / "src/trackmaniarl",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("tests", "scripts"):
        shutil.copytree(
            source / name,
            target / "src" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copyfile(source / name, target / "src" / name)
    shutil.copytree(
        baseline / "runtime",
        target / "runtime",
        ignore=shutil.ignore_patterns("preview-work", "parity-20261021"),
    )
    (target / "results").mkdir(mode=0o700)
    runtime = target / "runtime"
    for name in ("assets.json", "deployment.json"):
        path = runtime / name
        data = json.loads(path.read_bytes())
        data["runtime_root"] = str(runtime)
        path.write_text(json.dumps(data, indent=2) + "\n")
    config = json.loads((baseline / "runtime/config.json").read_bytes())["runtime"]
    config.update(
        runtime_root=str(runtime),
        trace_dir=str(target / "results"),
        work_dir=str(runtime / "build/server-probe/parity-work"),
        asset_manifest=str(runtime / "assets.json"),
        deployment_manifest=str(runtime / "deployment.json"),
        deployment_manifest_sha256=hashlib.sha256(
            (runtime / "deployment.json").read_bytes()
        ).hexdigest(),
        allow_native=True,
    )
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    uv = str(Path.home() / ".local/bin/uv")
    subprocess.run([uv, "venv", "--python", "3.12", str(target / "venv")], check=True)
    subprocess.run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(target / "venv/bin/python"),
            "--cache-dir",
            str(target / "uv-cache"),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy",
            "unsafe-best-match",
            "-r",
            str(baseline / "package/preview200/requirements-runtime-cpu.txt"),
            "pytest",
            "psutil",
        ],
        check=True,
    )
    (target / "setup.json").write_text(
        json.dumps(
            {
                "baseline": str(baseline),
                "source": str(source),
                "target": str(target),
                "sdk": "1.2.14",
                "private_assets_copied_only": True,
                "baseline_modified": False,
                "runtime_qualified_by_copy": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
