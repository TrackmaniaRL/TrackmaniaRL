"""Build a new read-only append witness; never edit an existing native SDK."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
out=Path(sys.argv[1]);cfg=json.loads((out/'environment.json').read_text());private=Path(cfg['private'])
base=Path('/home/kuba/tmrl-full-qr-20261022-011');original=base/'terminal-boundary-build583/source/tools/search/forkshim'
build=private/'observer-work-v2/forkshim';shutil.copytree(original,build,ignore=shutil.ignore_patterns('target'))
shutil.copytree(original.parent/'forkoracle',build.parent/'forkoracle',ignore=shutil.ignore_patterns('target'))
cargo=(build/'Cargo.toml').read_text();cargo=cargo.split('[dev-dependencies]')[0]+'\n[workspace]\n';(build/'Cargo.toml').write_text(cargo)
file=build/'src/post_observe/waypoint_observe.rs';source=file.read_text();before=hashlib.sha256(source.encode()).hexdigest()
old='''        if read(*g.add(11),&mut data[448..480]) { mask|=512; }'''
extra='''        if read(*g.add(11),&mut data[448..480]) { mask|=512; }
        // Actual append locations for any supported CP count, read-only.
        for (pointer,header,last,address) in [(*g.add(6),416usize,480usize,496usize),(*g.add(11),448usize,488usize,504usize)] {
            let count=u32::from_le_bytes(data[header+12..header+16].try_into().unwrap()) as usize;
            if count==0 || count>64 { fail(819) }
            let selected=need(pointer.checked_add(16+(count-1)*8),819);
            if !read(selected,&mut data[last..last+8]) { fail(819) }
            w64(&mut data,address,selected);
        }'''
if source.count(old)!=1:raise RuntimeError('Pinned writer insertion site changed')
source=source.replace(old,extra).replace('b"TMWP0001"','b"TMWP0002"').replace('w32(&mut data,12,1);','w32(&mut data,12,2);')
file.write_text(source)
tool=Path('/home/kuba/tmrl-rea-tools-20261021-001');bin_dir=tool/'rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin'
env=dict(os.environ,CARGO_BUILD_JOBS='2',CARGO_TARGET_DIR=str(private/'observer-target'),CARGO_HOME=str(tool/'cargo'),RUSTUP_HOME=str(tool/'rustup'),PATH=str(bin_dir)+':'+os.environ.get('PATH',''))
subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(build/'Cargo.toml')],env=env,check=True,timeout=120)
target=Path(cfg['runtime_root'])/'build/forkoracle/multi1161/libforkshim.so';target.parent.mkdir(parents=True);shutil.copyfile(private/'observer-target/release/libforkshim.so',target)
manifest_path=private/'assets-multi.json';oldmanifest=json.loads(Path(cfg['manifest']).read_text());relative=str(target.relative_to(cfg['runtime_root']));oldmanifest['roles']['shim']=relative
oldmanifest['assets'][relative]=dict(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(oldmanifest,indent=2));cfg.update(manifest=str(manifest_path),manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),multi_waypoint_append_witness=True)
(out/'environment-multi.json').write_text(json.dumps(cfg,indent=2))
(out/'observer-build.json').write_text(json.dumps(dict(status='BUILT_NOT_QUALIFIED',original_source_sha256=before,modified_source_sha256=hashlib.sha256(source.encode()).hexdigest(),shim_sha256=oldmanifest['assets'][relative]['sha256'],input_or_physics_writes_added=False,measurement='two actual last append destinations and addresses, record schema TMWP0002'),indent=2))
for name in ('environment.py','multi_goal.py'):shutil.copyfile(Path(__file__).parent/name,Path(cfg['source'])/name)
print(json.dumps(cfg))
