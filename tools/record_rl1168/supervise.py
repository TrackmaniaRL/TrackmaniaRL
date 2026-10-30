"""Outer pidfd cohort owns trainer descendants and the absolute deadline."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    root=Path(sys.argv[1]);launch=json.loads((root/'launch.json').read_text())
    prep=Path(launch['preparation']);plan=json.loads((prep/'plan.json').read_text())
    sys.path.insert(0,plan['backend_config']['source'])
    from trackmaniarl.experiments.full_qr.campaign_processes import CampaignProcessCohort
    from trackmaniarl.trackmania.native_headless._runtime.ownership import process_identity
    from tools.record_rl1168.train import write
    until=datetime.fromisoformat(launch['deadline_utc'].replace('Z','+00:00'))
    started=time.monotonic();budget=(until-datetime.now(timezone.utc)).total_seconds()
    private=Path('/home/kuba/tmrl-full-qr-20261022-011')/(root.name+'-supervision')
    def stop():
        run=root/'run'
        if run.exists():(run/'STOP').touch(exist_ok=True)
    cohort=CampaignProcessCohort(private,stop)
    report=dict(status='starting',parent=process_identity(os.getpid()),deadline_utc=launch['deadline_utc'])
    try:
        if budget<1800:raise ValueError('insufficient new budget')
        for name,digest in json.loads((root/'source-sha256.json').read_text()).items():
            if hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen source changed')
        report['trainer']=cohort.start('trainer',[sys.executable,'-B','-m','tools.record_rl1168.train',
            str(prep),launch['qualification'],str(root/'run'),'--until-utc',launch['deadline_utc']],cwd=root/'source')
        report['status']='running'
        while not cohort.exited():
            if time.monotonic()-started>=budget-30 or(root/'STOP').exists():
                stop();report.update(status='stopped',stop_reason='absolute_deadline_or_operator');break
            for path,gib in (('/mnt/c',10),('/mnt/f',4),(str(private),4)):
                if shutil.disk_usage(path).free<gib*2**30:raise RuntimeError('disk reserve '+path)
            write(root/'supervisor-state.json',dict(report,elapsed_seconds=time.monotonic()-started))
            time.sleep(3)
        report['exits']=cohort.exited()
        if report['exits']=={'trainer':0}:
            completion=json.loads((root/'run/completion.json').read_text())
            report['status']='completed' if completion['status']=='completed' else 'failed'
            report['trainer_status']=completion['status']
        elif report['exits']:
            report.update(status='failed',error='trainer exited unsuccessfully')
    except BaseException as error:
        report.update(status='failed',error=repr(error))
    finally:
        report['closure']=cohort.close(grace_seconds=20)
        if any(report['closure'][k] for k in ('remainingOwned','cleanupErrors','forced')):report['status']='failed'
        report['elapsed_seconds']=time.monotonic()-started
        write(root/'supervision.json',report)
    if report['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
