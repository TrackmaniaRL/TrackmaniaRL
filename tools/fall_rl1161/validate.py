from pathlib import Path
import json,os,shutil,subprocess,sys,time
out=Path(sys.argv[1]);tag=sys.argv[2];file=Path(sys.argv[3]);cfg=json.loads((out/'environment-source.json').read_text());private=Path(cfg['private']);sys.path.insert(0,cfg['source'])
from trackmaniarl.trackmania.native_headless._runtime.supervisor import spawn_supervised
from trackmaniarl.trackmania.native_headless._runtime.validator import parse_records
from trackmaniarl.trackmania.native_headless._runtime.shapes import ParseRecordsSpec
from environment import sha
work=private/tag;work.mkdir();(work/'UserData/Maps').mkdir(parents=True);(work/'UserData/Replays').mkdir();(work/'CommonData').mkdir();(work/'Logs').mkdir()
manifest=json.loads(Path(cfg['manifest']).read_text());runtime=Path(cfg['runtime_root']);server=runtime/manifest['roles']['server'];mp=runtime/manifest['roles']['map']
os.symlink(server,work/'TrackmaniaServer');os.symlink(server.parent/'Packs',work/'Packs');shutil.copyfile(mp,work/'UserData/Maps/fall.Map.Gbx');shutil.copyfile(file,work/'UserData/Replays/candidate.Replay.Gbx')
report=dict(status='FAILED',source_sha256=sha(file),map_sha256=sha(mp),server_sha256=sha(server));process=None
try:
 with (work/'stderr').open('xb') as error:
  process=spawn_supervised(['./TrackmaniaServer','/nodaemon','/validatepath=.'],cwd=work,env={k:v for k,v in os.environ.items() if k!='LD_PRELOAD' and not k.startswith(('FK_','FKSHIM_'))},stderr=error,evidence=work/'closure.json',timeout=30)
  raw=process.stdout.read();(work/'stdout').write_bytes(raw)
  end=time.monotonic()+30
  while process.poll() is None:
   if time.monotonic()>end:raise RuntimeError('Bounded native validation')
   time.sleep(.05)
  if process.poll()!=0:raise RuntimeError('Native validator exit failed')
 records=parse_records(ParseRecordsSpec(text=raw.decode(),expected_files={'candidate.Replay.Gbx'},expected_map_uid='v96wCr7mP5xEGxNLSnZC6DAp4se',mode='plain',allow_inherited_array_close=False))
 report.update(status='NATIVE_VALIDATOR_READ',records=records)
finally:
 if process is not None:
  report['closure']=process.stop(timeout=5);process.close_control()
 (out/(tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
