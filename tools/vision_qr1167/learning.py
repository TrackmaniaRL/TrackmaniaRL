"""Bounded replay and the repository's Double QR-DQN learner."""
from collections import deque
import random
import numpy as np
import torch
from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.contracts import RiskSpec, ValuePhase
from tools.vision_qr1167.model import build_model, ACTIONS
from tools.vision_qr1167.returns import driving_return


class Learning:
    def __init__(self, seed=1167, device='cuda', capacity=5000):
        self.rng = random.Random(seed)
        self.replay = deque(maxlen=capacity)
        self.replay_episode_ids = deque(maxlen=capacity)
        self.episode_counter = 0
        self.current_episode = None
        self.learner = DiscreteValueLearner(build_model(), seed=seed,
            learning_rate=1e-4, target_update_interval=500,
            execution={'device': device, 'precision': 'float32'})
        self.learner.setup({'seed': seed})
        self.transitions = 0
        self.retention = None
        self.pending_retention = None
        self.action_ids = None

    def configure_action_ids(self, action_ids):
        """An explicit action-space choice for a fresh learner only."""
        if self.learner.update_count:
            raise ValueError('action-space changes require a fresh learner')
        self._restore_action_ids(action_ids)

    def extend_actions_from_retention(self):
        """Admit only newly demonstrated actions from a verified data extension."""
        if self.retention is None:
            raise ValueError('verified retention extension required')
        if self.action_ids is None:
            return []
        additions=sorted(set(self.retention.actions.tolist())-set(self.action_ids))
        if not additions:
            return []
        if (not self.pending_retention or self.retention.parent_sha256!=self.pending_retention['dataset_sha256']):
            raise ValueError('new actions require a verified extension of checkpoint data')
        self._restore_action_ids(tuple(sorted(set(self.action_ids)|set(additions))))
        return additions

    def _restore_action_ids(self, action_ids):
        ids = None if action_ids is None else tuple(action_ids)
        if ids is not None and (not ids or len(ids)!=len(set(ids)) or
                any(type(i) is not int or not 0<=i<len(ACTIONS) for i in ids)):
            raise ValueError('invalid policy action IDs')
        self.action_ids = ids
        # The SDK applies this mask to Double-Q bootstrap selection as well.
        self.learner.policy_action_ids = ids

    @torch.no_grad()
    def action(self, observation, epsilon):
        if self.rng.random() < epsilon:
            ids = getattr(self, 'action_ids', None)
            return self.rng.choice(ids) if ids is not None else self.rng.randrange(len(ACTIONS))
        model = self.learner.model
        tensor = torch.as_tensor(observation[None], device=self.learner.device)
        features = model.encode_frames(tensor)
        support = model.support(features, ValuePhase.EVALUATE)
        values = model.expected_all_actions(features, support, RiskSpec())
        ids = getattr(self, 'action_ids', None)
        if ids is not None:
            indices = torch.tensor(ids, device=values.device)
            return ids[int(values.index_select(-1, indices).argmax(-1).item())]
        return int(values.argmax(-1).item())

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        if self.current_episode is None:
            raise RuntimeError('begin a real episode before adding transitions')
        self.replay.append((observation.copy(), action, reward, next_observation.copy(), terminated, truncated))
        self.replay_episode_ids.append(self.current_episode)
        self.transitions += 1

    def begin_episode(self):
        self.episode_counter += 1
        self.current_episode = self.episode_counter

    def configure_retention(self, root):
        from tools.vision_qr1167.retention import Retention
        retention = Retention(root, self.learner.device)
        if self.pending_retention:
            if (retention.sha256 != self.pending_retention['dataset_sha256'] and
                    retention.parent_sha256 != self.pending_retention['dataset_sha256']):
                raise ValueError('retention dataset changed on resume')
            retention.weight = self.pending_retention['weight']
            retention.steps = self.pending_retention['steps']
        self.retention = retention

    def update(self, batch_size=32):
        if self.pending_retention and self.retention is None:
            raise RuntimeError('configure preserved auxiliary imitation before resuming RL')
        if len(self.replay) < batch_size:
            return None
        rows = list(self.replay)
        ids = list(self.replay_episode_ids)
        if len(rows) != len(ids):
            raise ValueError('replay episode provenance misaligned')
        indices = self.rng.sample(range(len(rows)), batch_size)
        samples = [driving_return(rows,ids,i) for i in indices]
        obs, act, rew, nxt, done, discounts, horizons = zip(*samples)
        done_tensor = torch.tensor(done, dtype=torch.bool)
        batch = TrainingBatch(data={}, observations=torch.from_numpy(np.stack(obs)),
            actions=torch.tensor(act, dtype=torch.long), rewards=torch.tensor(rew, dtype=torch.float32),
            next_observations=torch.from_numpy(np.stack(nxt)), terminated=done_tensor,
            truncated=torch.zeros(batch_size, dtype=torch.bool),
            bootstrap_discounts=torch.tensor(discounts, dtype=torch.float32),
            transition_ids=list(range(batch_size)), metadata={'gamma': .997,
                'return_horizon_max':5,'return_lengths':list(horizons),
                'return_semantics':'finite driving attempt; failures terminal; explicit per-sample discounts'})
        metrics, _ = self.learner.update(batch)
        if self.retention is not None:
            metrics = dict(metrics, **self.retention.step(self.learner))
        if not all(torch.isfinite(p).all() for p in self.learner.model.parameters()):
            raise FloatingPointError('non-finite online parameters')
        return dict(metrics)

    def checkpoint(self):
        return dict(learner=self.learner.state_dict(), replay=list(self.replay),
                    replay_episode_ids=list(self.replay_episode_ids),episode_counter=self.episode_counter,
                    return_semantics='finite-driving-v1',
                    rng=self.rng.getstate(), transitions=self.transitions,
                    action_catalog=ACTIONS, observation='real-camera-4x84x84',
                    policy_action_ids=getattr(self, 'action_ids', None),
                    auxiliary_imitation=self.retention.state_dict() if self.retention else self.pending_retention)

    def restore(self, checkpoint):
        if checkpoint.get('return_semantics') not in (None,'finite-driving-v1'):
            raise ValueError('unsupported return semantics')
        if tuple(checkpoint['action_catalog']) != ACTIONS or checkpoint['observation'] != 'real-camera-4x84x84':
            raise ValueError('checkpoint action/observation contract mismatch')
        self.learner.load_state_dict(checkpoint['learner'])
        self._restore_action_ids(checkpoint.get('policy_action_ids'))
        self.replay.extend(checkpoint['replay'])
        episode_ids=checkpoint.get('replay_episode_ids',[None]*len(checkpoint['replay']))
        if len(episode_ids)!=len(checkpoint['replay']):
            raise ValueError('checkpoint replay episode provenance misaligned')
        self.replay_episode_ids.extend(episode_ids)
        self.episode_counter=checkpoint.get('episode_counter',0)
        self.current_episode=None
        self.rng.setstate(checkpoint['rng'])
        self.transitions = checkpoint['transitions']
        self.pending_retention = checkpoint.get('auxiliary_imitation')
