"""Explicit optional supervised retention alongside real QR updates.

This is auxiliary imitation, not an environment reward or an RL transition.
It decays with RL updates, so it does not permanently lock the expert actions.
"""
import hashlib,json
import io
from pathlib import Path
import numpy as np
import torch
from trackmaniarl.algorithms.value_based.objectives import DemonstrationMarginObjective,ValueObjectiveContext
from trackmaniarl.models.contracts import ValuePhase,RiskSpec


def validated_parent(receipt, data):
    parent_sha=receipt.get('parent_dataset_sha256')
    if parent_sha is None:
        return None
    payload=Path(receipt['parent_archive']).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=parent_sha:
        raise ValueError('retention parent archive changed')
    with np.load(io.BytesIO(payload),allow_pickle=False) as parent:
        count=receipt['parent_rows']
        if count!=len(parent['actions']) or set(parent.files)!=set(data.files):
            raise ValueError('retention parent structure changed')
        for key in parent.files:
            if data[key].dtype!=parent[key].dtype or not np.array_equal(data[key][:count],parent[key]):
                raise ValueError('retention extension replaced base rows')
    admission=Path(receipt['addition_receipt']).read_bytes()
    if hashlib.sha256(admission).hexdigest()!=receipt['addition_receipt_sha256']:
        raise ValueError('retention addition receipt changed')
    added_receipt=json.loads(admission)
    added_bytes=Path(receipt['addition_archive']).read_bytes()
    if (added_receipt['status']!='ADMITTED_TEACHER_WINDOWS' or added_receipt['teacher_goal_ms']!=17601 or
            hashlib.sha256(added_bytes).hexdigest()!=added_receipt['dataset_sha256'] or
            added_receipt['dataset_sha256']!=receipt['addition_dataset_sha256']):
        raise ValueError('retention addition not admitted')
    with np.load(io.BytesIO(added_bytes),allow_pickle=False) as addition:
        if set(addition.files)!=set(data.files):
            raise ValueError('retention addition structure changed')
        for key in addition.files:
            if data[key].dtype!=addition[key].dtype or not np.array_equal(data[key][count:],addition[key]):
                raise ValueError('retention extension contains unadmitted rows')
    return parent_sha


def phase_balanced_indices(phases, step, batch_size):
    if not phases or any(len(phase)==0 for phase in phases):
        raise ValueError('empty demonstration phase')
    serial=np.arange(step*batch_size,(step+1)*batch_size)
    return np.array([phases[int(s%len(phases))][int(s//len(phases))%len(phases[int(s%len(phases))])]
                     for s in serial],dtype=np.int64)


class Retention:
    def __init__(self, root, device, weight=.1):
        root=Path(root)
        receipt=json.loads((root/'receipt.json').read_text())
        payload=(root/'demonstrations.npz').read_bytes()
        if receipt['status']!='ADMITTED_TEACHER_WINDOWS' or hashlib.sha256(payload).hexdigest()!=receipt['dataset_sha256']:
            raise ValueError('retention dataset provenance')
        data=np.load(root/'demonstrations.npz',allow_pickle=False)
        self.parent_sha256=validated_parent(receipt,data)
        self.observations=torch.from_numpy(data['observations']).to(device)
        self.actions=torch.from_numpy(data['actions']).long().to(device)
        bins=data['race_ms']//2000
        self.phases=[np.flatnonzero(bins==key) for key in np.unique(bins)]
        self.sha256=receipt['dataset_sha256']
        self.weight=weight
        self.steps=0

    def step(self, learner, batch_size=16):
        # Deterministic cycling; no changes to replay sampling's RNG.
        indices=torch.as_tensor(phase_balanced_indices(self.phases,self.steps,batch_size),device=learner.device)
        model=learner.model
        features=model.encode_frames(self.observations[indices])
        support=model.support(features,ValuePhase.EVALUATE)
        expected=model.expected_all_actions(features,support,RiskSpec())
        objective=DemonstrationMarginObjective(weight=self.weight/(1+learner.update_count/5000))
        mask = None
        if learner.policy_action_ids is not None:
            mask = torch.zeros(expected.shape[-1], dtype=torch.bool, device=expected.device)
            mask[list(learner.policy_action_ids)] = True
        context=ValueObjectiveContext(expected,self.actions[indices],
            torch.ones(batch_size,dtype=torch.bool,device=learner.device),dict(demo_flags=[True]*batch_size),mask)
        loss=objective.loss(context)
        learner.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        learner.optimizer.step()
        self.steps+=1
        return {'loss/auxiliary_imitation':float(loss.detach()),
                'auxiliary_imitation_updates':self.steps}

    def state_dict(self):
        return dict(dataset_sha256=self.sha256,weight=self.weight,steps=self.steps,
                    sampling='uniform 2-second demonstration phases; clock is sampler metadata, never a policy input')
