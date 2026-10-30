"""Fresh BC point-mass prior with provenance-bound, real off-policy replay.

This deliberately starts a new RL learner/optimizer. It never labels the
previous learner's updates as updates of the new model.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tools.vision_qr1167.learning import Learning
from tools.vision_qr1167.model import ACTIONS
from tools.vision_qr1167.quantile_initialization import initialize_point_mass
from trackmaniarl.models.contracts import RiskSpec, ValuePhase


def load(path):
    raw = path.read_bytes()
    return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('bc', type=Path)
    parser.add_argument('closed_campaign', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument('--steering-only', action='store_true', help='17 steering values; requires all demonstrations to use full throttle without braking')
    controls.add_argument('--supported-actions', action='store_true', help='17 full-throttle steering values plus every observed demonstration maneuver')
    args = parser.parse_args()
    torch.set_num_threads(2)
    closed = json.loads((args.closed_campaign/'completion.json').read_text())
    owner = json.loads((args.closed_campaign/'windows-launch.json').read_text())
    if closed['cleanupErrors']:
        raise ValueError('previous cleanup incomplete')
    subprocess.run(['powershell','-NoProfile','-Command',
        f"$p=Get-Process -Id {int(owner['pid'])} -ErrorAction SilentlyContinue;"
        f"if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq '{int(owner['birthUtcTicks'])}'){{exit 9}};exit 0"], check=True)
    prior, prior_sha = load(args.closed_campaign/'latest.pt')
    if prior_sha != closed['checkpoint_sha256']:
        raise ValueError('closed replay checkpoint changed')
    bc, bc_sha = load(args.bc/'initialization.pt')
    receipt = json.loads((args.bc/'result.json').read_text())
    if receipt['status']!='BC_INITIALIZATION_ONLY' or receipt['policy_sha256']!=bc_sha or receipt['rl_updates']!=0:
        raise ValueError('BC provenance mismatch')
    data_receipt = json.loads((args.dataset/'receipt.json').read_text())
    data_bytes = (args.dataset/'demonstrations.npz').read_bytes()
    if hashlib.sha256(data_bytes).hexdigest()!=receipt['dataset_sha256'] or receipt['dataset_sha256']!=data_receipt['dataset_sha256']:
        raise ValueError('BC dataset changed')
    if tuple(prior['action_catalog'])!=ACTIONS or prior['observation']!=bc['observation']:
        raise ValueError('replay observation/action mismatch')
    rows = prior['replay']
    for obs, action, reward, nxt, done, trunc in rows:
        if obs.dtype!=np.uint8 or nxt.dtype!=np.uint8 or obs.shape!=(4,84,84) or nxt.shape!=(4,84,84):
            raise ValueError('replay camera contract')
        if not 0<=action<len(ACTIONS) or not np.isfinite(reward):
            raise ValueError('replay action/reward contract')
    learning = Learning(device='cpu')
    learning.restore(bc)
    if learning.learner.update_count!=0 or learning.learner.optimizer.state or learning.replay:
        raise ValueError('initialization already contains RL state')
    data = np.load(io.BytesIO(data_bytes), allow_pickle=False)
    observations = torch.from_numpy(data['observations'])
    model = learning.learner.model
    def values():
        features = model.encode_frames(observations)
        support = model.support(features, ValuePhase.EVALUATE)
        return model.expected_all_actions(features, support, RiskSpec())
    with torch.no_grad():
        before = values()
        initialize_point_mass(model.head)
        after = values()
        if not torch.allclose(before, after, atol=1e-5, rtol=1e-5) or not torch.equal(before.argmax(-1),after.argmax(-1)):
            raise ValueError('point-mass conversion changed supervised decisions')
    learning.learner.target_model.load_state_dict(model.state_dict())
    if args.steering_only:
        learning.configure_action_ids(tuple(range(17)))
    elif args.supported_actions:
        learning.configure_action_ids(tuple(sorted(set(range(17)) | set(map(int,data['actions'])))))
    if learning.action_ids is not None and not set(map(int,data['actions'])).issubset(learning.action_ids):
        raise ValueError('action space excludes demonstrated maneuvers')
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('nonfinite initialization')
    admitted_rows = [row for row in rows if learning.action_ids is None or row[1] in learning.action_ids]
    learning.replay.extend(admitted_rows)
    learning.replay_episode_ids.extend([None]*len(admitted_rows))
    learning.transitions = prior['transitions']
    args.output.mkdir(exist_ok=False)
    checkpoint = learning.checkpoint()
    checkpoint['initialization_provenance'] = dict(bc_sha256=bc_sha,replay_checkpoint_sha256=prior_sha,new_rl_optimizer=True)
    torch.save(checkpoint,args.output/'initialization.pt')
    report = dict(status='BC_INITIALIZATION_ONLY',rl_updates=0,bc_updates=receipt['bc_updates'],
        max_abs_q=float(after.abs().max()),max_mean_q_change=float((after-before).abs().max()),
        quantile_prior='point mass at supervised mean Q',fresh_rl_optimizer=True,
        imported_real_replay_rows=len(admitted_rows),excluded_unavailable_action_rows=len(rows)-len(admitted_rows),imported_transitions=learning.transitions,
        previous_model_rl_updates=closed['learner_updates'],previous_model_weights_resumed=False,
        policy_action_ids=learning.action_ids,
        control_space=('17 full-throttle steering values plus all demonstrated maneuvers; learned selection throughout lap' if args.supported_actions else 'learned steering with constant full throttle' if args.steering_only else 'learned steering, throttle and brake'),
        replay_checkpoint_sha256=prior_sha,bc_checkpoint_sha256=bc_sha,
        dataset_sha256=receipt['dataset_sha256'],
        policy_sha256=hashlib.sha256((args.output/'initialization.pt').read_bytes()).hexdigest(),
        generalization_claim=False,autonomous_finish_verified=False)
    (args.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
