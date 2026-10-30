"""Append independently admitted teacher windows without replacing base rows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('base',type=Path)
    parser.add_argument('addition',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    receipts=[];arrays=[]
    for root in (args.base,args.addition):
        receipt=json.loads((root/'receipt.json').read_text())
        payload=(root/'demonstrations.npz').read_bytes()
        if receipt['status']!='ADMITTED_TEACHER_WINDOWS' or receipt['teacher_goal_ms']!=17601 or hashlib.sha256(payload).hexdigest()!=receipt['dataset_sha256']:
            raise ValueError('unadmitted teacher data')
        receipts.append(receipt)
        with np.load(root/'demonstrations.npz',allow_pickle=False) as data:
            arrays.append({key:data[key] for key in data.files})
    if receipts[0]['teacher_source_sha256']!=receipts[1]['teacher_source_sha256'] or set(arrays[0])!=set(arrays[1]):
        raise ValueError('source or array contract changed')
    combined={key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    args.output.mkdir(exist_ok=False)
    np.savez_compressed(args.output/'demonstrations.npz',**combined)
    report=dict(receipts[0],dataset_sha256=hashlib.sha256((args.output/'demonstrations.npz').read_bytes()).hexdigest(),
        samples=len(combined['actions']),parent_dataset_sha256=receipts[0]['dataset_sha256'],
        rejected={key:sum(r['rejected'].get(key,0) for r in receipts) for key in set().union(*(r['rejected'] for r in receipts))},
        source_receipts=[dict(collection_sha256=r['source_sha256'],dataset_sha256=r['dataset_sha256']) for r in receipts],
        padded_initial_windows=sum(r.get('padded_initial_windows',0) for r in receipts),
        initial_padding=receipts[1].get('initial_padding'),
        parent_archive=str((args.base/'demonstrations.npz').resolve()),parent_rows=len(arrays[0]['actions']),
        addition_dataset_sha256=receipts[1]['dataset_sha256'],addition_archive=str((args.addition/'demonstrations.npz').resolve()),
        addition_receipt=str((args.addition/'receipt.json').resolve()),
        addition_receipt_sha256=hashlib.sha256((args.addition/'receipt.json').read_bytes()).hexdigest(),
        early_windows_before=int((arrays[0]['race_ms']<2000).sum()),early_windows_added=int((arrays[1]['race_ms']<2000).sum()),
        merge_semantics='All base rows preserved as an exact prefix; separately admitted windows appended; not a new autonomous result')
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
