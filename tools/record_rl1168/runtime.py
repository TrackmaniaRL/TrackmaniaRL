"""Copy immutable runtime assets into a fresh campaign-owned guarded layout."""

import hashlib
import json
import shutil
from pathlib import Path


def isolate_runtime(config, private):
    config = dict(config)
    private = Path(private)
    old_root = Path(config["runtime_root"])
    manifest_bytes = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != config.get("manifest_sha256"):
        raise ValueError("source asset manifest SHA mismatch")
    manifest = json.loads(manifest_bytes)
    root = private / "runtime"
    root.mkdir(exist_ok=False)
    if shutil.disk_usage(private).free < 5 * 2**30:
        raise RuntimeError("ext4 reserve plus runtime copy budget")
    server = Path(manifest["roles"]["server"]).parent
    shutil.copytree(old_root / server, root / server)
    for name, item in manifest["assets"].items():
        source = old_root / name
        destination = root / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError("source asset changed: " + name)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("runtime copy hash mismatch")
    manifest["runtime_root"] = str(root)
    target = private / "assets.json"
    target.write_text(json.dumps(manifest, indent=2))
    config.update(
        private=str(private),
        runtime_root=str(root),
        manifest=str(target),
        manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    return config
