"""Completed native leases must not span arbitrary learner-side waits."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.unit.models.test_full_qr_environment import setup
from trackmaniarl.trackmania.native_headless.full_qr.native_provider import NativeFullQrProvider
from trackmaniarl.trackmania.native_headless.protocols import NativeLifecycleError


def test_terminal_releases_driver_before_transition_is_returned():
    environment, provider, _ = setup()
    environment.reward_config = replace(environment.reward_config, minimum_finish_steps=1, finish_progress=0.0001)
    provider.finish_episode = Mock()
    environment.reset()
    environment.step(41)
    provider.finish_episode.assert_not_called()
    provider.finish = True
    _, _, terminated, _, info = environment.step(41)
    assert terminated and info['race_time_ms'] == 100
    provider.finish_episode.assert_called_once_with()


def test_release_failure_cannot_be_a_successful_training_transition():
    environment, provider, _ = setup()
    environment.reward_config = replace(environment.reward_config, minimum_finish_steps=1, finish_progress=0.0001)
    provider.finish_episode = Mock(side_effect=RuntimeError('closure not proven'))
    environment.reset()
    provider.finish = True
    with pytest.raises(NativeLifecycleError, match='no transition returned'):
        environment.step(41)
    assert provider.closed == 1


def test_single_episode_driver_closes_without_waiting_for_next_reset():
    provider = object.__new__(NativeFullQrProvider)
    provider._session = object()
    provider.driver_episode_limit = provider._driver_episodes = 1
    provider.events = SimpleNamespace(release_episode=Mock())
    provider._finish_session = Mock(side_effect=lambda reason: setattr(provider, '_session', None))
    provider.finish_episode()
    provider.events.release_episode.assert_called_once_with()
    provider._finish_session.assert_called_once_with('episode_complete')
    # Any learner-side delay after this point leaves no native deadline alive.
    provider.finish_episode()
    assert provider._session is None
    assert provider._finish_session.call_count == 1
