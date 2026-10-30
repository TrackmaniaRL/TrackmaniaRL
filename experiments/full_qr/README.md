# Full QR engineering tools

The installed implementation lives in:

- `trackmaniarl.trackmania.native_headless.full_qr`: game frames, qualified
  event sources, historical features/rewards, environment factory and retention;
- `trackmaniarl.experiments.full_qr`: model execution, owned campaign processes,
  wall-clock budgets and construction of a separate V8 source deployment.

See [the full QR guide](../../readme/headless-full-qr.md) and
[measured results](../../docs/benchmarks/full-qr-headless.md).

This directory retains optional engineering tools and offline comparison helpers.
`native/` contains instrumentation/build sources, not a bundled game, shim,
checkpoint or accepted replay. They require explicitly supplied, pinned evidence
and toolchain inputs. Run repository tools as Python modules from the checkout.
`scripts/diagnostic_inputs.py` rejects absent or noncanonical evidence paths.

The optional `openplanet/Headless*` diagnostic probes are repository-only
research tools; the Python wheel does not install them. They and the Windows
comparison helpers require deliberately configured local inputs. They are
separate from training and are not needed by a qualified headless deployment.
Missing observations, stale pointers, EOF and unqualified constants remain
failures rather than substitute training data.

Original laboratory notes, raw receipts, native assets and checkpoints are
retained outside the distribution. Historical home paths are not installation
defaults. Source layout changes do not transfer qualification by themselves:
the new deployment must pass its source/reference and native gates.

## Explicit diagnostic inputs

Historical native tools use only the inputs they need from this table. Each
configured path must already exist, be absolute and canonical. Their original
hash, framing and source checks remain mandatory after path resolution.

| Variable | Caller-supplied input |
|---|---|
| `TRACKMANIARL_NATIVE_CONFIG` | Pinned predecessor runtime configuration |
| `TRACKMANIARL_NATIVE_RECEIPT` | Original build receipt |
| `TRACKMANIARL_NATIVE_REFERENCE` | Actual finite reference capture |
| `TRACKMANIARL_NATIVE_BASELINE` | Qualified predecessor source/runtime root |
| `TRACKMANIARL_NATIVE_SOURCE` | Explicit experimental source build |
| `TRACKMANIARL_NATIVE_EVIDENCE` | Qualified engineering evidence directory |
| `TRACKMANIARL_RUST_TOOLS` | Existing reviewed Rust toolchain/cache root |

`input_consumer_native_replay` instead requires its existing `--v6-config` and
`--receipt` arguments explicitly. No personal checkout is an installation default.

Opt-in Windows comparisons require `TRACKMANIARL_COMPARISON_ROOT` and
`TRACKMANIARL_PROBE_CONFIG`, plus the `psutil` diagnostic dependency. For example,
run a tool with `uv run --with psutil python -m scripts.headless_windows_timing`.
The comparison root supplies the existing preflight/controller lease helpers;
it is trusted application code, not an arbitrary downloaded evidence file.
The standalone input-profile and UI-lease tools require `--controller-lease`.
Callers injecting `NoArtificialLagLease` may provide the existing lease callback,
or deliberately configure `TRACKMANIARL_CONTROLLER_LEASE`. All original mutex,
foreground, TTL and rollback checks remain active. These utilities are separate
from headless training and are never invoked by importing the SDK.
