from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
BASE=Path('/home/kuba/tmrl-full-qr-20261022-011')
FALL=Path('/mnt/f/AITrackmania/experiments/fall2026-05-ghosts-0749e38ff0f6')
ORIGINAL=BASE/'waypoint-runtime002'
HERE=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def main():
 name='fall-rl1161-'+time.strftime('%Y%m%d-%H%M%S',time.gmtime())
 out=FALL/name;private=BASE/name;out.mkdir();private.mkdir(mode=0o700)
 source=private/'source';shutil.copytree(BASE/'shortcut-multi-source1094/trackmaniarl',source/'trackmaniarl',ignore=shutil.ignore_patterns('__pycache__'))
 for f in ('environment.py',):shutil.copyfile(HERE/f,source/f)
 build=private/'tape-build';build.mkdir();shutil.copyfile(HERE/'tape.rs',build/'main.rs')
 (build/'Cargo.toml').write_text('''[workspace]
[package]
name="fall-tape1161"
version="0.1.0"
edition="2021"
[dependencies]
gbx={path="/home/kuba/tmrl-full-qr-20261022-011/ghost-build393/source/gbx"}
serde_json="=1.0.145"
[[bin]]
name="fall-tape1161"
path="main.rs"
''')
 toolchain=Path('/home/kuba/tmrl-rea-tools-20261021-001')
 bin_dir=toolchain/'rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin'
 env=dict(os.environ,CARGO_BUILD_JOBS='2',CARGO_TARGET_DIR=str(private/'cargo-target'),CARGO_HOME=str(toolchain/'cargo'),RUSTUP_HOME=str(toolchain/'rustup'),PATH=str(bin_dir)+':'+os.environ.get('PATH',''))
 subprocess.run(['cargo','build','--offline','--release','--manifest-path',str(build/'Cargo.toml')],env=env,check=True,timeout=120)
 view=private/'runtime';roles=dict(server='build/server-probe/linux-validate-map2-001/TrackmaniaServer',driver='build/driver-long150/tmre-reset-driver',
 shim='build/forkoracle/terminal583/libforkshim-impact.so',map='analysis/tmrl-shortcut/assets-001/tmrl-test.Map.Gbx',tape='analysis/tmrl-shortcut/fall1161/INPUTONLY-neutral.Replay.Gbx')
 for role in ('server','driver','shim'):
  p=view/roles[role];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ORIGINAL/roles[role],p);p.chmod(0o755)
 shutil.copytree((ORIGINAL/roles['server']).parent/'Packs',(view/roles['server']).parent/'Packs')
 mp=view/roles['map'];mp.parent.mkdir(parents=True);shutil.copyfile(FALL/'map.Map.Gbx',mp)
 tape=view/roles['tape'];tape.parent.mkdir(parents=True)
 replay=FALL/'model-ray-projection-17030-plus-audio-v4-recorder/model-21.758.Replay.Gbx'
 subprocess.run([str(private/'cargo-target/release/fall-tape1161'),str(replay),str(ORIGINAL/'analysis/tmrl-shortcut/long150-controls001/restored-inputs.Ghost.Gbx'),str(tape),str(out/'source-controls-TRAINING-ONLY.json')],check=True,timeout=30)
 manifest=dict(schema='fall05.native-physics17-assets/v1',runtime_root=str(view),roles=roles,
  assets={name:dict(bytes=(view/name).stat().st_size,sha256=sha(view/name)) for name in roles.values()},
  map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',full_V5_or_UI_qualification_claimed=False)
 manifest_path=private/'assets.json';manifest_path.write_text(json.dumps(manifest,indent=2))
 config=dict(runtime_root=str(view),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),report=str(out),private=str(private),source=str(source))
 (out/'environment.json').write_text(json.dumps(config,indent=2));(private/'environment.json').write_text(json.dumps(config,indent=2))
 (out/'source-manifest.json').write_text(json.dumps([dict(path=str(p),sha256=sha(p)) for p in sorted(source.rglob('*')) if p.is_file()],indent=2))
 (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_NATIVE_QUALIFICATION_REQUIRED',config=config,source_replay_sha256=sha(replay),source_replay_expected_native_ms=21758,training_started=False),indent=2))
 print(json.dumps(config))
if __name__=='__main__':main()
