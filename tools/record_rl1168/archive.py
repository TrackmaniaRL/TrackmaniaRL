"""Lossless archival of closed traces created by this campaign only."""

import hashlib
import json
import tarfile
from pathlib import Path

from tools.record_rl1168.closure import native_closure_clean


def archive_closed_trace(private, trace, destination):
    private = Path(private).resolve()
    trace = Path(trace).resolve()
    destination = Path(destination)
    if not trace.is_relative_to(private / "traces") or trace == private / "traces":
        raise ValueError("trace outside this campaign")
    close = json.loads((trace / "native-supervisor.json").read_text())
    if not native_closure_clean(close):
        raise ValueError("trace does not have clean closure")
    files = sorted(p for p in trace.rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(trace) for p in trace.rglob("*")):
        raise ValueError("trace link escapes campaign")
    hashes = {str(p.relative_to(trace)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with (
        destination.open("xb") as output,
        tarfile.open(fileobj=output, mode="w:gz", compresslevel=1) as archive,
    ):
        for p in files:
            archive.add(p, arcname=str(p.relative_to(trace)), recursive=False)
    with tarfile.open(destination, "r:gz") as archive:
        verified = {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive
        }
    if verified != hashes:
        raise ValueError("trace archive failed full SHA roundtrip")
    receipt = {
        "archive": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "members": hashes,
        "source": str(trace),
        "closed_identity": close,
        "roundtrip_verified": True,
    }
    destination.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    # Only remove exact verified files under the NEW campaign, retaining archive.
    for p in files:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(trace))]:
            raise ValueError("closed trace unexpectedly changed")
    for p in files:
        p.unlink()
    for p in sorted(trace.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if p.is_dir():
            p.rmdir()
    trace.rmdir()
    return receipt
