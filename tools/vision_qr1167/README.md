# tmrl-test vision QR preparation (1167)

The bounded campaign launcher and real-camera QR-DQN runner are implemented.
The authoritative live status is `STUDY/vision-qr1167/campaign-*/state.json`;
the existence of code or a launcher receipt alone is not evidence of RL updates.

## Implemented

- Real-camera encoder: four 84x84 grayscale frames, 2,015,968 parameters.
- Existing SDK QR-DQN head/strategy: 64 quantiles and 68 absolute actions,
  covering 17 steering levels and all four accelerator/brake combinations.
  Control is available over the whole lap, not only its final segment.
- Training sampler: 50% full starts, 50% uniformly sampled supplied snapshots;
  initial steering perturbation in [-0.12, 0.12], lasting 1–5 physics ticks.
- Distance reward paid only for previously unreached progress, time penalty,
  goal bonus, corridor hint, jump rejection and terminal guard.
- Evaluation sampler always uses the actual map start with no snapshot or
  perturbation. Curriculum goal times are never accepted as lap records.

## Runtime contract and limitations

1. TICK steps via a virtual controller. Each nonterminal policy transition must
   contain exactly five fresh 10ms native steps. Actual native inputs are logged;
   controller input has a measured pipeline delay and steering is gamepad-scaled.
2. Real HUD pixel readback checks the rendered clock. TICK's native telemetry
   clock label precedes the capture/HUD clock by 10ms. Host capture time is not
   used as proof. HUD, clock and TICK trajectory hints are excluded from policy.
3. Cached revision teleport did not establish live physics stepping. The runner
   instead executes a real teacher prefix, then pauses and releases playback.
   Its measured handoff time varies slightly; no hidden-state restore parity is
   claimed. Half of episodes use an actual reset with teacher disabled.
4. Local ordered arc-distance projection prevents jumping to distant crossing
   segments. The reference was executed in this game session with finish 17601ms.
5. Retained PPO weights are incompatible with this CNN QR model. This campaign
   starts a new vision learner, with no claim of BC initialization. Checkpoints
   retain online/target/optimizer/RNG and the bounded replay buffer.
6. Training includes random initial steering and suffixes. Its goals are NOT
   certified records. A separate frozen-policy full-start evaluation, trace
   audit and export validation are still required for any record candidate.

## Live observations, 29 October 2026

TICK launcher default port 46322 failed with address-in-use. Explicit loopback
port 46342 started the launcher and data service successfully. tmrl-test UID
was verified. Effective game speed zero was acknowledged. Capture and restore
commands succeeded; restore produced state-replay and revision-rewind events.
Keyboard steps failed, but controller steps advanced live physics. Seven camera
samples independently matched native capture clocks; short prefix-handoff
rollouts passed exact tick and rendered clock checks. The original failures are
retained in separate report directories. The unsigned render-marker experiment
was rejected by Openplanet; signing settings were not changed. HUD readback is
used instead.

Evidence and retained checkpoints live in STUDY/vision-qr1167. The old residual
campaign's original FAILED status is preserved: it exited after the requested
operator STOP, not because the new CNN experiment failed.

Tests: `python -B -m unittest tools.vision_qr1167.test_curriculum tools.vision_qr1167.test_reference -v`.
The CNN forward/backward test used synthetic tensors only; it is not a driving
evaluation or an RL update on real experience.
