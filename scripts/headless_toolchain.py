"""Download isolated official REA/Ghidra/build tools; never reconfigure an agent."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url: str) -> bytes:
    with urlopen(
        Request(url, headers={"User-Agent": "TrackmaniaRL-local-research"}), timeout=60
    ) as response:
        return response.read()


def download(root: Path, specification: tuple[str, str, str]) -> dict[str, str | int]:
    name, url, expected = specification
    algorithm = "sha256"
    path = root / name
    digest = hashlib.new(algorithm)
    with (
        urlopen(
            Request(url, headers={"User-Agent": "TrackmaniaRL-local-research"}), timeout=60
        ) as response,
        path.open("xb") as stream,
    ):
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            stream.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("download digest differs: " + name)
    return {
        "path": str(path),
        "url": url,
        "algorithm": algorithm,
        "digest": expected,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    root = Path(sys.argv[1])
    if (
        sys.platform != "linux"
        or not root.is_absolute()
        or root.exists()
        or str(root).startswith("/mnt/")
    ):
        raise ValueError("fresh ext4 tools root required")
    root.mkdir(mode=0o700)
    node_sums = fetch("https://nodejs.org/dist/v24.22.1/SHASUMS256.txt").decode()
    node_hash, node_name = next(
        line.split() for line in node_sums.splitlines() if line.endswith("linux-x64.tar.xz")
    )
    jdk = json.loads(
        fetch(
            "https://api.adoptium.net/v3/assets/version/21.0.12.1%2B1"
            "?architecture=x64&image_type=jdk&os=linux&vendor=eclipse"
        )
    )[0]["binary"]["package"]
    ghidra = json.loads(
        fetch(
            "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/tags/Ghidra_12.1.4_build"
        )
    )["assets"][0]
    rust_url = "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
    rust_hash = fetch(rust_url + ".sha256").decode().split()[0]
    jobs = [
        (root, node_name, "https://nodejs.org/dist/v24.22.1/" + node_name, node_hash),
        (root, jdk["name"], jdk["link"], jdk["checksum"]),
        (root, ghidra["name"], ghidra["browser_download_url"], ghidra["digest"].split(":")[1]),
        (root, "rustup-init", rust_url, rust_hash),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda args: download(args[0], args[1:]), jobs))
    (root / "downloads.json").write_text(json.dumps(receipts, indent=2) + "\n")
    for row in receipts:
        path = Path(row["path"])
        if path.name.endswith((".tar.xz", ".tar.gz")):
            with tarfile.open(path) as archive:
                archive.extractall(root, filter="data")
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    if not (root / entry.filename).resolve().is_relative_to(root):
                        raise ValueError("zip path escapes tool root")
                archive.extractall(root)
    node = root / node_name.removesuffix(".tar.xz")
    jdk_home = next(p for p in root.iterdir() if p.is_dir() and (p / "bin/javac").is_file())
    # ZipFile does not restore executable bits from Unix archives.
    ghidra_home = root / "ghidra_12.1.4_PUBLIC"
    for path in ghidra_home.rglob("*"):
        if path.is_file() and (
            path.name in ("analyzeHeadless", "launch.sh", "decompile", "sleigh")
            or path.suffix == ".sh"
        ):
            path.chmod(path.stat().st_mode | 0o100)
    environment = dict(
        os.environ,
        PATH=str(node / "bin") + ":" + os.environ["PATH"],
        npm_config_cache=str(root / "npm-cache"),
    )
    subprocess.run(
        [
            str(node / "bin/npm"),
            "install",
            "--prefix",
            str(root / "rea"),
            "--ignore-scripts",
            "--no-audit",
            "--no-fund",
            "rea-agents@6.7.0",
        ],
        env=environment,
        check=True,
    )
    rust = root / "rustup-init"
    rust.chmod(0o700)
    subprocess.run(
        [
            str(rust),
            "-y",
            "--no-modify-path",
            "--profile",
            "minimal",
            "--default-toolchain",
            "1.99.1",
        ],
        env=dict(environment, CARGO_HOME=str(root / "cargo"), RUSTUP_HOME=str(root / "rustup")),
        check=True,
    )
    (root / "toolchain.json").write_text(
        json.dumps(
            {
                "node": str(node),
                "java": str(jdk_home),
                "ghidra": str(ghidra_home),
                "rea": str(root / "rea/node_modules/rea-agents"),
                "cargo": str(root / "cargo"),
                "rustup": str(root / "rustup"),
            },
            indent=2,
        )
        + "\n"
    )
    print((root / "toolchain.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
