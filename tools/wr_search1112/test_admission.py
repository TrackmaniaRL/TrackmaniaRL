"""Admission must preserve the exact failed campaign, never turn it into PASS."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from .run import check_prior_documents


def documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint = {'sha256': 'e06c33110eacb3d5eaad00592210adaa422c94926dfca8fc200a3312da747a3c'}
    exits = {'learner': 0, 'actor-000': 1, 'actor-001': 0, 'actor-002': 0}
    closure: dict[str, list[Any]] = {'remainingOwned': [], 'cleanupErrors': [], 'forced': []}
    done = {'phase': 'failed', 'status': 'FAILED_OR_INCOMPLETE', 'exits': exits,
            'closure': closure, 'result': {'checkpoint': checkpoint}}
    audit = {'status': 'PASS_PRESERVATION_AUDIT_OF_FAILED_RUN_NOT_TRAINING_SUCCESS',
             'exits': exits.copy(), 'closure': deepcopy(closure), 'remaining_exact_identities': [],
             'finite_final_tensors': {'online': 90, 'target': 90, 'optimizers': 264},
             'final_checkpoint': checkpoint.copy(), 'power_request': {'phase': 'released'}}
    windows = {'old_owner_identity_gone': True, 'old_power_guard_identity_gone': True}
    return done, audit, windows


def test_accept_only_preserved_specific_closed_failure() -> None:
    done, audit, windows = documents()
    before = deepcopy(done)
    check_prior_documents(done, audit, windows)
    assert done == before
    assert done['phase'] == 'failed'


@pytest.mark.parametrize(
    'problem', ['status', 'exit', 'cleanup', 'finite', 'sha', 'power', 'owner'],
)
def test_reject_incomplete_or_different_failure(problem: str) -> None:
    done, audit, windows = documents()
    if problem == 'status':
        done['phase'] = 'completed'
    elif problem == 'exit':
        audit['exits']['actor-001'] = 1
    elif problem == 'cleanup':
        done['closure']['remainingOwned'] = [{'pid': 999}]
    elif problem == 'finite':
        audit['finite_final_tensors']['optimizers'] = 0
    elif problem == 'sha':
        audit['final_checkpoint']['sha256'] = 'different'
    elif problem == 'power':
        audit['power_request']['phase'] = 'held'
    else:
        windows['old_owner_identity_gone'] = False
    with pytest.raises(RuntimeError):
        check_prior_documents(done, audit, windows)


def test_only_search_omits_unused_terminal_time_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    from trackmaniarl.experiments.shortcut_macro import actual_goal_reward
    from trackmaniarl.trackmania.native_headless.full_qr.environment import FullQrEnvironment
    from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError
    from trackmaniarl.trackmania.reward_config import RewardConfig
    from trackmaniarl.trackmania.reward_types import RewardResult

    from .environment import SequenceSearchEnvironment

    raw = RewardResult(100, True, 'finished')
    with pytest.raises(NativeLifecycleError, match='sourced final reward interval'):
        actual_goal_reward(raw, RewardConfig(), (21350, 21283))
    monkeypatch.setattr(FullQrEnvironment, '_observe_and_reward', lambda *_: ({}, raw))
    search = object.__new__(SequenceSearchEnvironment)
    assert search._observe_and_reward(None)[1] is raw  # type: ignore[arg-type]


def test_search_runtime_paths_are_private_ext4_and_archives_remain_in_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import run

    monkeypatch.setattr(run, 'read', lambda _: {'runtime': {'trace_dir': '/mnt/f/old'}})
    monkeypatch.setattr(run, 'sha', lambda _: 'a' * 64)
    study = Path('/mnt/f/study')
    root = study / 'local-search1113'
    trial = root / 'execution/trial-0000'
    config = run.search_config(study, root, trial)
    assert config['runtime']['trace_dir'] == str(run.BASE / root.name / trial.name / 'traces')
    assert config['runtime']['work_dir'].endswith('local-search1113-trial-0000')
    assert config['trace_archive_root'] == str(trial / 'closed-traces')
    assert config['runtime']['timeout_s'] == 150
