"""Read already-closed autonomous archives; no game, simulation or input changes."""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

root=Path(sys.argv[1]).resolve()
if root.exists():raise FileExistsError('Fresh diagnostic directory required')
study=root.parent
sys.path.insert(0,'/mnt/c/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania')
from tools.model_ghost1117.extract import extract
from tools.model_ghost1117.visual_state import visual_records

physics=extract(study) if len(sys.argv)<3 else json.loads((Path(sys.argv[2])/'physics.json').read_text())
archive=Path(physics['provenance']['archive'])
pid=physics['provenance']['native_pid']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=physics['provenance']['archive_sha256']:raise ValueError('Bound archive hash')
with tarfile.open(archive,'r:gz') as source:
    stream=source.extractfile(f'trace/posts/vehicle-{pid}.bin')
    if stream is None:raise RuntimeError('Actual visual recording missing')
    raw=stream.read()
records=visual_records(raw,pid)
origin=physics.get('native_origin_ms',2200)
assert all(f['t']+origin in records for f in physics['frames'])
root.mkdir();(root/'source').mkdir()
(root/'recorded-visual.bin').write_bytes(raw)
(root/'physics.json').write_text(json.dumps(physics,allow_nan=False))
layout=Path('/home/kuba/tmrl-full-qr-20261022-011/ghost-build393/source/fk/fk/src/vislayout.rs')
shutil.copyfile(layout,root/'source/vislayout.rs')
shutil.copyfile(Path(__file__).with_name('main.rs'),root/'source/main.rs')
manifest=dict(status='OFFLINE_ONLY_PREPARED_NOT_A_VALIDATED_GHOST',
    archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    raw_sha256=hashlib.sha256(raw).hexdigest(),physics_sha256=hashlib.sha256((root/'physics.json').read_bytes()).hexdigest(),
    native_pid=pid,records=len(records),new_simulations=0,source_goal_ms=physics['goal_ms'],
    layout_sha256=hashlib.sha256(layout.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((root/'source/main.rs').read_bytes()).hexdigest())
(root/'provenance.json').write_text(json.dumps(manifest,indent=2))
cargo='''[workspace]
[package]
name="native-ghost1140"
version="0.1.0"
edition="2021"
[dependencies]
gbx={path="/home/kuba/tmrl-full-qr-20261022-011/ghost-build393/source/gbx"}
ghost={path="/home/kuba/tmrl-full-qr-20261022-011/ghost-build393/source/ghost"}
serde_json="=1.0.145"
[[bin]]
name="native-ghost1140"
path="source/main.rs"
'''
(root/'Cargo.toml').write_text(cargo)
print(json.dumps(manifest))
