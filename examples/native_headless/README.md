# Native headless project example

This opt-in example composes the experimental native environment with a
terminal-aware collector, replay, sampler and QR-DQN learner. It demonstrates
application integration; the default training CLI and generated projects keep
their existing contracts.

Read the [backend guide](../../readme/native-headless.md) before running native
code. From a development checkout, install dependencies with `uv` and inspect
the example's explicit options:

```bash
uv sync --group dev
uv run python -m examples.native_headless.train --help
```

With the exact external assets already provisioned, run this explicit bounded
attempt from a Linux/WSL checkout. Point `RUNTIME` at an already provisioned
qualified deployment. Output and work paths must be fresh and outside the
source checkout; the native workspace remains inside the runtime root.

```bash
RUNTIME="$HOME/tm-runtime"
DEPLOYMENT="$RUNTIME/deployment.json"
DEPLOYMENT_SHA=$(sha256sum "$DEPLOYMENT" | cut -d ' ' -f1)
uv run python -m examples.native_headless.train \
  --execute-native-model \
  --runtime-root "$RUNTIME" \
  --asset-manifest "$RUNTIME/assets.json" \
  --deployment-manifest "$DEPLOYMENT" \
  --deployment-manifest-sha256 "$DEPLOYMENT_SHA" \
  --out "$HOME/native-headless-runs/attempt-001/output" \
  --work "$RUNTIME/build/server-probe/native-headless-example-001" \
  --train-episodes 1 \
  --seed 1729
```

The example permits one to three fresh training episodes. It uses one CPU
thread and a 44-input, 78-action, 32-quantile QR-DQN model. Greedy evaluation runs
before and after training with separate episode IDs and no replay ingestion or
update credit. Its `final.pt` is a new example artifact; there is no resume flag.
The pinned prebuilt driver requires work directories beneath the runtime root's
`build/server-probe/` tree. The output directory above remains outside the source
checkout.

Training earns one update per four accepted training transitions. Fractional
credit carries between episodes; short episodes retain pending credit until
replay has at least 64 real rows. Updates happen after the native episode is
closed, and the actor receives a fresh policy snapshot for each episode. With
one complete 799-transition training episode, this gives 199 updates and a
three-transition remainder. The two 799-transition evaluation episodes add no
training credit.

Supply your external asset manifest through `NativeRuntimeConfig`. The
[example manifest](assets.example.json) lists the fixed layout and expected
hashes, without supplying those files. Its hashes
bind the native executable, driver, shim, map, tape, reference and cold-state
evidence. These assets are not shipped with TrackmaniaRL. A different manifest
root does not make an unsupported driver portable. The portable profile requires
`--deployment-manifest` and `--deployment-manifest-sha256`, alongside the asset
manifest, binding a qualified driver and Packs. Use fresh output/work directories
for each attempt. The [native sources](../../native/README.md) are included;
fresh builds need qualification before they can replace a pinned binary.

The example uses the exact 44-feature/78-action profile. A normal reset starts at
native clock 2230; the administrative warmup is excluded from replay. At the
39950-unit duration limit, 799 full N5 decisions end with an actual final
44-feature observation and truncation.
The learner can bootstrap this endpoint. Native clocks remain raw native units,
separate from wall time, race time and a physical finish event.

A qualified natural terminal can contain actual native76 but no final feature44.
The collector keeps that absence explicit, and the ragged target path evaluates
only rows with a real bootstrap observation. It does not invent a zero vector or
reuse the previous observation. Inconclusive evidence raises before replay and
update credit. Evaluation episodes have separate IDs and do not enter replay or
spend training credit. Close the environment in `finally`, including failures.

`NativeHeadlessEnvironmentFactory` is an environment component seam, not a promise
that the generic collector handles missing terminal44. The explicit collector
and target handling here are part of the example's contract. This is not a
replacement for the standard SAC33/Openplanet configuration.

Start fresh. Historical research checkpoints and replay files have their own
fingerprints and are not an implicit warm start or resume input for this port.
Keep all raw successes and failures outside package/distribution inputs. A
successful bounded run establishes the checked scope of that run; it does not
establish client parity, faster training, a learned finish or generalization.
