"""Build and bind an immutable separately qualified V8 SDK deployment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

_SCHEMA = "tmrl.record-rl1168.v8-deployment/v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> dict[str, str]:
    package = root / "trackmaniarl"
    if package.is_symlink() or not package.is_dir():
        raise ValueError("a real trackmaniarl deployment directory is required")
    result = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError("deployment symlinks are forbidden")
        if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo"):
            raise ValueError("deployment must not contain bytecode caches")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha(path)
    receipt = root / "source-deployment.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise ValueError("V8 source deployment receipt is required")
    result["source-deployment.json"] = _sha(receipt)
    if "trackmaniarl/__init__.py" not in result:
        raise ValueError("V8 package initializer missing")
    return result


def _canonical_directory(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("canonical absolute deployment directory required")
    if not path.is_dir():
        raise ValueError("deployment must be a directory")
    return path


def _expected_checkout_files() -> dict[str, str]:
    # Evaluate the checked-in transformer in a separate interpreter. Importing it
    # in this process would preload the checkout's non-V8 trackmaniarl package.
    source = Path(__file__).resolve().parents[2]
    script = r"""
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import trackmaniarl
if Path(trackmaniarl.__file__).resolve() != root / "trackmaniarl/__init__.py":
    raise RuntimeError("wrong checkout package imported")
from trackmaniarl.experiments.full_qr.runtime_profile import _REPLACEMENTS, transform_source
package = root / "trackmaniarl"
files = {}
for path in sorted(package.rglob("*")):
    if path.is_symlink():
        raise ValueError("checkout SDK symlinks are forbidden")
    if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
        continue
    relative = path.relative_to(root).as_posix()
    raw = path.read_bytes()
    if relative in _REPLACEMENTS:
        raw = transform_source(relative, raw.decode("utf-8")).encode("utf-8")
    files[relative] = hashlib.sha256(raw).hexdigest()
if not set(_REPLACEMENTS).issubset(files):
    raise ValueError("V8 transformation sources missing")
print(json.dumps(files, sort_keys=True))
"""
    process = subprocess.run(
        [sys.executable, "-B", "-c", script, str(source)],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(process.stdout)


def activate_from_environment() -> dict[str, str]:
    """Verify all deployment files and import the SDK only from that deployment."""
    raw = os.environ.get("TMRL_V8_DEPLOYMENT", "")
    manifest_raw = os.environ.get("TMRL_V8_MANIFEST", "")
    expected = os.environ.get("TMRL_V8_MANIFEST_SHA256", "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("qualified V8 deployment manifest SHA256 required")
    deployment = _canonical_directory(raw)
    manifest = Path(manifest_raw)
    if (
        not manifest.is_absolute()
        or manifest.is_symlink()
        or manifest.resolve(strict=True) != manifest
    ):
        raise ValueError("canonical absolute V8 manifest required")
    if not manifest.is_file() or _sha(manifest) != expected:
        raise ValueError("V8 deployment manifest SHA mismatch")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != _SCHEMA or data.get("deployment") != str(deployment):
        raise ValueError("V8 manifest identity mismatch")
    files = data.get("files")
    if not isinstance(files, dict) or files != _files(deployment):
        raise ValueError("V8 deployment files changed")
    sdk_files = {key: value for key, value in files.items() if key.startswith("trackmaniarl/")}
    if sdk_files != _expected_checkout_files():
        raise ValueError("V8 deployment does not match the current checkout")
    receipt = json.loads((deployment / "source-deployment.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "tmrl.full-qr-source-deployment/v1"
        or receipt.get("protocol") != "full-qr-current150/v8"
        or receipt.get("destination") != str(deployment)
    ):
        raise ValueError("separate V8 deployment receipt required")
    package_root = deployment / "trackmaniarl"
    for name, loaded in tuple(sys.modules.items()):
        if name != "trackmaniarl" and not name.startswith("trackmaniarl."):
            continue
        module_file = getattr(loaded, "__file__", None)
        if not module_file or not Path(module_file).resolve().is_relative_to(package_root):
            raise RuntimeError(
                "trackmaniarl was already imported outside the verified V8 deployment"
            )
    # The deployment is immutable: imports must not create __pycache__ in it.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(deployment))
    package = importlib.import_module("trackmaniarl")
    module_file = Path(package.__file__).resolve(strict=True)
    if module_file != deployment / "trackmaniarl/__init__.py":
        raise RuntimeError("trackmaniarl import did not come from verified V8 deployment")
    return {
        "deployment_path": str(deployment),
        "deployment_manifest_sha256": expected,
        "imported_package": str(module_file),
    }


def create(source: Path, deployment: Path, manifest: Path) -> None:
    source = source.resolve(strict=True)
    if deployment.exists() or manifest.exists():
        raise ValueError("fresh deployment and manifest required")
    if not deployment.is_absolute() or not manifest.is_absolute():
        raise ValueError("absolute deployment and manifest paths required")
    if manifest.parent.resolve(strict=True) != manifest.parent or manifest.is_relative_to(
        deployment
    ):
        raise ValueError("manifest must be outside the deployment in a canonical directory")
    sys.path.insert(0, str(source))
    import trackmaniarl
    from trackmaniarl.experiments.full_qr.runtime_profile import build_source_deployment

    if Path(trackmaniarl.__file__).resolve() != source / "trackmaniarl/__init__.py":
        raise RuntimeError("source SDK import did not come from the selected checkout")
    build_source_deployment(source, deployment)
    deployment = _canonical_directory(str(deployment))
    files = _files(deployment)
    data = {"schema": _SCHEMA, "deployment": str(deployment), "files": files}
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
    print(
        json.dumps(
            {
                "deployment": str(deployment),
                "manifest": str(manifest),
                "sha256": _sha(manifest),
                "file_count": len(files),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    create(args.source, args.deployment, args.manifest)


if __name__ == "__main__":
    main()
