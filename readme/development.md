# Developing TrackmaniaRL

## Repository setup

Requirements are Git, Python 3.12 and
[uv](https://docs.astral.sh/uv/). From a clone:

```bash
git clone https://github.com/TrackmaniaRL/TrackmaniaRL.git
cd TrackmaniaRL
uv sync --group dev
uv run trackmaniarl --help
```

`uv sync` creates the managed environment and installs this checkout in editable
mode. Do not activate the environment manually and do not install dependencies
with `pip`.

Run the focused local gate before and after a change:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```

This runs the maintained regression suite. The explicit full-suite command is:

```bash
uv run poe test-full
```

For a quick iteration, run a relevant test first, then the full gate:

```bash
uv run pytest tests/integration/runtime/test_core_runtime.py -q
uv run pytest tests/integration/runtime/test_core_runtime.py::test_resolved_run_writes_manifest_and_smoke_checkpoint -q
```

## Repository map

```text
trackmaniarl/              published library
  core/                    stable contracts, data, replay, spec and runtime
  algorithms/              learners and optimization utilities
    value_based/            unified scalar/quantile learner, targets and losses
  models/                  reusable neural-network building blocks
    encoders/               frame-only MLP/CNN encoders
    temporal/               Identity, GRU and portable Mamba cores
    heads/                  scalar, quantile, actor and critic heads
    strategies/             scalar/fixed/random/learned value support
  builtins/                supported component catalogue
  distributed/             actor/learner protocol and durability
  trackmania/              game-only adapter
  observability/           manifests, events and artifacts
  experiments/             evaluation/study orchestration
  project/                 `trackmaniarl init` templates
tests/                     deterministic unit and integration tests
readme/                    user and developer guides
docs/diagrams/             reproducible architecture diagrams and previews
```

`trackmaniarl init <project-name>` creates an application project outside the
library boundary. Generated projects and their artifacts are ignored source-tree
state, not release contents.

## Change workflow

### README media and package metadata

The root README is also the PyPI package description. Keep its links absolute
HTTPS URLs. Repository documentation uses `blob/main/` and media uses
`raw.githubusercontent.com/TrackmaniaRL/TrackmaniaRL/main/` so bumping the package
version before creating a release tag does not break the published page.
These links follow the current documentation and artwork; they are not frozen
release snapshots. Historical benchmark evidence keeps its own provenance.

Before publishing, verify that every new linked file has been pushed to `main`.
Check image URLs with an unauthenticated request and confirm the response is an
image, not an HTML page or a Git LFS pointer. The documentation tests check local
targets and prevent version-derived media links from returning.

Build the package and check the wheel's `METADATA` description as well as the
repository README. Changing the repository does not update the description of an
already published PyPI release; the correction ships with a new package release.

### Code changes

1. Decide whether the behavior is reusable library code or application-specific
   experiment code. Prefer the generated agent project for the latter.
2. Identify the owning package from the architecture document.
3. Add or update the smallest explicit contract only if existing contracts
   cannot express the behavior.
4. Implement the component without import-time I/O or optional dependency
   requirements in the core path.
5. Add a deterministic test beside the nearest existing contract tests.
6. Run formatting, strict typing and the maintained regression suite.
7. Update user/developer documentation and editable diagram sources whenever a
   public flow, ownership boundary or checkpoint contract changes.
8. Regenerate each changed diagram from its `.spec.json`, validate the
   `.excalidraw` scene and visually inspect its SVG/PNG preview at documentation
   width.
9. For game changes, run `trackmaniarl track check` and the bounded smoke test
   on Windows with Trackmania before release.

## Adding a dependency

Use `uv add <package>` for a required dependency or place the package in the
correct optional-dependency group. Keep optional integrations optional at
import time. Commit both `pyproject.toml` and `uv.lock`, then verify:

```bash
uv lock --check
uv sync --locked --group dev
```

Dependencies used only by generated projects belong in the generated
`pyproject.toml`, not automatically in the library runtime.

The distributed rollout journal owns one SQLite connection and lock per
instance. Evaluation delivery and promotion operations live in
`distributed/journal_evaluation.py` and share that connection and lock; in
particular, claiming an evaluation result remains part of the rollout append
transaction. Append acquires a SQLite write transaction before checking its
receipt, so separate journal connections serialize the idempotency decision.
Pruning also reserves the write transaction before checking the frontier, so
concurrent prunes cannot move it backward after rows have been deleted. These
narrow guarantees do not make multiple active coordinators on one journal safe:
checkpoint validation, recovery, profile allocation and evaluation delivery
require a single coordinator owner.
Initial creation and supported schema migration also reserve the write
transaction before inspecting the schema, allowing independent openers to
observe one complete result. A transient busy response while switching a new
database to WAL is retried for up to five seconds; persistent lock errors remain
visible to the caller.
Journal schema and supported checkpoint formats are unchanged by this split.

## Adding a public component

A public component should have:

- one clear responsibility and full type annotations.
- an existing `trackmaniarl.core` protocol, or a justified minimal new one.
- a stable import path documented in the SDK guide or built-in catalogue.
- deterministic contract and state round-trip tests.
- no hidden global configuration or mandatory network tracker.
- checkpoint state for everything required to resume correctly.

Value-model additions must also declare their representation contract, validate
encoder/temporal/head/strategy dimensions, cover `[B,...]` and `[B,T,...]` and
prove that selected-action paths do not materialize unnecessary all-action
tensors. A learned value strategy needs optimizer-isolation and finite-difference
gradient tests.

Temporal-core additions need `unroll`, `step`, initial-state and burn-in tests.
Backend substitutions may change kernels but must not silently change model
parameters or architecture fingerprints.

Add new algorithms or encoders as importable, opt-in blocks. Compare
one variable at a time against an identical seeded baseline before promoting a
new default.

## Validation levels

| Level | Command | Purpose |
| --- | --- | --- |
| configuration | `uv run trackmaniarl validate run.yaml` | imports components and performs a synthetic update without the game |
| unit/integration | `uv run poe test` | critical configuration, learning, runtime and distribution regression paths |
| game connection | `uv run trackmaniarl track check --config run.yaml` | verifies three 33-field frames, session protocol, active UID and readiness |
| bounded live gate | `uv run trackmaniarl smoke run.yaml --transitions 100` | real actor/learner path, policy refresh and checkpoint |
| training | `uv run trackmaniarl train run.yaml` | full configured run |

For behavior cloning, `validate` uses the offline-supervised validation hook.
The bounded workflow additionally includes deterministic split tests, exact
`bc-latest.pt` resume and a real `bc-benchmark` closed-loop release gate.

`validate` imports configured Python components, so it is safe for the game but
not a sandbox for untrusted projects.

## Release checklist

Before creating a tag, run the local quality gate and inspect both distributions:

```bash
uv run poe fmt
uv run poe types
uv run poe test
uv build --clear
uv run python scripts/check_distribution.py
```

The tag workflow repeats the source gate on Ubuntu and Windows. Both platforms
export the locked development environment with Torch pruned, then run the gate
in an isolated environment with the exact CPU-only Torch wheel. This preserves
the locked versions and hashes for every other dependency without downloading
the CUDA dependency tree. Generated application projects select their own
accelerator source. The workflow then builds exactly once on Ubuntu. The
resulting wheel and source archive, `SHA256SUMS` and SPDX 2.3 JSON SBOM are uploaded
together as the single `release-dist` artifact. Ubuntu and Windows download and
verify that artifact, including its checksums, wheel CLI and generated-project
resolution. Only after both verifiers pass does the final job attach and locally
verify GitHub SLSA provenance and SBOM attestations, create and locally verify
PEP 740 publish attestations and publish the same wheel and source-archive
bytes to PyPI. The publish job must not check out the repository or rebuild the
package.

After PyPI accepts the verified artifacts, a separate least-privilege job creates
the GitHub Release. It attaches the canonical distribution, checksums, SBOM,
README logo and full-drive GIF. The PyPI description uses the versioned raw GIF
URL. Its bounded size lets GitHub return `image/gif` for inline rendering.

All release actions are pinned to full commit SHAs, checkout credentials remain
disabled and the publish job receives only read-only repository metadata plus
the OIDC and attestation write permissions it needs. Update those pins and the
concrete uv/Syft versions as an explicit, reviewed maintenance change.

After release, independently download each PyPI distribution and verify both
GitHub predicates and the PyPI-hosted publish attestation. Substitute the exact
repository, release commit and distribution filename:

```bash
gh attestation verify "$ARTIFACT" --repo "$REPOSITORY" --signer-workflow "$REPOSITORY/.github/workflows/release.yml" --source-digest "$COMMIT"
gh attestation verify "$ARTIFACT" --repo "$REPOSITORY" --signer-workflow "$REPOSITORY/.github/workflows/release.yml" --source-digest "$COMMIT" --predicate-type https://spdx.dev/Document/v2.3
uvx --from pypi-attestations==0.0.30 pypi-attestations verify pypi --repository "https://github.com/$REPOSITORY" "pypi:<distribution-filename>"
```

### Optional extended Windows soak evidence

Run this extended validation on a real Windows TrackMania host with the first-party
`OpenPlanetEnvironmentFactory`, the signed plugin in School Mode and the map UID
and geometry used by deterministic evaluation. Use it for changes to runtime,
transport or checkpoint durability. It is not required for every library
release. Do not substitute fake actors or the bounded smoke test.

1. Start a normal local run and keep the same code revision and `run_id` for the
   whole soak. Wait for a `train/checkpoint_completed` event before stopping it
   gracefully.
2. Resume that exact run with `uv run trackmaniarl resume run.yaml
   artifacts/<run-id>/checkpoints/<checkpoint>.pt`. Continue until a newer
   checkpoint completes, then stop the learner and actor. The sum of `elapsed_s`
   maxima across all process segments, excluding time between processes, must
   be at least four hours when performing the extended soak.
3. Run `uv run trackmaniarl benchmark run.yaml
   artifacts/<run-id>/checkpoints/<newer-checkpoint>.pt`. The configured release
   thresholds must pass and the resulting `evaluation.json` must contain no
   trial with a `telemetry_error` or `controller_error`.
4. Stop every TrackManiaRL process completely. Never run the verifier against
   an active artifact directory, because it hashes evidence, benchmark and
   checkpoint files.
5. Run `uv run python scripts/verify_soak.py artifacts/<run-id>` and retain the
   generated `artifacts/<run-id>/soak-report.json` with the release evidence.

The verifier fails closed on malformed inputs and checks the immutable manifest,
per-attempt Windows environment snapshots and complete JSONL event stream. Its
report binds the run, process segments, accepted 64-character run fingerprint,
stable actor IDs and fresh session IDs.
adds the observed runtime. Proves the resumed policy version against a completed
checkpoint. Records monotonic transition and WAL checkpoint frontiers. Hashes
the resume and post-resume checkpoint artifacts. Binds the final benchmark and
its artifact hash to the post-resume checkpoint SHA-256. Checks every benchmark
trial for controller/telemetry errors. And requires no runtime, transport,
telemetry or checkpoint failure events. It never loads checkpoint contents or
starts, stops or connects to TrackMania.

Finally, verify a clean wheel installation and update the changelog for
user-visible behavior and the security policy/audit when a trust boundary
changes.
