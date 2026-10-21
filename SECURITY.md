# Security policy

## Reporting a vulnerability

Do not publish suspected vulnerabilities, credentials, checkpoints, telemetry
captures or unredacted manifests in a public issue. Use a private
[GitHub security advisory](https://github.com/TrackmaniaRL/TrackmaniaRL/security/advisories/new)
and include:

- the affected TrackmaniaRL version and platform;
- the smallest safe reproduction or malformed input;
- the expected impact and required attacker access;
- whether the issue affects local, distributed or Trackmania operation.

Do not access other users' systems or data while investigating. The maintainer
will acknowledge a complete report, assess severity and coordinate a fix and
disclosure. Security fixes target the current `1.2` release line.

## Trust boundaries

- `run.yaml` is trusted executable configuration. Component paths import and
  instantiate Python objects; `validate`, `train`, `learner` and `actor` must
  only receive configurations and extension packages from trusted sources.
  `inspect-config` safely parses and lists every `class_path` without importing
  it, but it is an inspection aid rather than a sandbox or signature check.
- Checkpoints are data, but should still come from a trusted run. The default
  codec uses PyTorch `weights_only=True`; it rejects payloads requiring
  executable pickle globals. A custom `CheckpointCodec` defines its own trust
  boundary.
- RunSpec 2.0 resume verifies a complete architecture fingerprint. Partial
  warm-start deliberately accepts selected compatible tensors and therefore
  requires review of its generated match report. Neither mechanism proves the
  checkpoint's origin.
- Demonstrations and geometry use NumPy loading with `allow_pickle=False`.
  BC manifests hash demonstrations and split membership, but hashes do not
  authenticate an untrusted dataset. Size and semantic correctness must still
  be suitable for the run.
- Distributed gRPC uses a bearer token and loopback-only learner binding. The
  token authenticates but does not encrypt. Remote actors require an
  authenticated encrypted tunnel and a random token of at least 32 characters.
- OpenPlanet telemetry and session ports are localhost-only and are not an
  internet-facing API. Use the signed TrackmaniaRL Connect plugin from Plugin
  Manager in School Mode; the local session protocol cannot attest the plugin
  package's signature or installed version.
- Generated Trackmania projects explicitly pin `vgamepad`; the published
  `trackmania` extra does not install it. Its normal installer may install
  or repair the system ViGEmBus driver. Set `VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true`
  to skip that installer only when a compatible driver is already provisioned.

Never commit `.env`, API keys, distributed tokens, raw telemetry containing
personal data, or private checkpoints. Run manifests redact keys whose names
contain `key`, `token`, `secret` or `password`, but custom component payloads
must avoid placing secrets under misleading names.

## Publication privacy gate

Before publishing, inspect both the working tree and reachable Git history.
Deleting a file or replacing a local path in the current tree does not remove
earlier copies. Coordinate any history rewrite and credential rotation separately;
never force-push a rewritten history without maintainer agreement.

Run `uv run python scripts/check_distribution.py --dist-dir dist` against the
actual source archive and wheel. It rejects private file types, environment
files, cache directories, archive links, duplicate members, oversized members,
personal home paths and common credential signatures, including in metadata.
This is a regression guard, not proof that all secrets or identifying data are
absent. Review free-form configuration, uncommon credential formats, media
pixels, account/session identifiers and private URLs manually. Public license
attribution and repository identity are intentional publication metadata.

Use environment variables for credentials; `.env-example` contains empty values.
The library does not automatically load `.env`. Export only the variables needed
by the selected integration and keep tokens out of command-line arguments,
component kwargs, screenshots and shared run directories. Local manifests and
recording sidecars may contain paths and timestamps even when tokens are redacted.

## Dependency advisory disposition for 1.2.0

`PYSEC-2026-3447` affects source-distribution exclusion matching in setuptools
before 83.0.0. PyTorch 2.11.0 requires runtime `setuptools<82`, so that runtime
dependency still appears in the raw audit. It must not be used to build release
archives. Both this package and generated projects pin `setuptools==83.0.0`
in their isolated build requirements and CI builds with isolation enabled.

For 1.2.0, the disposition is **accepted for the Torch runtime only, mitigated
for package building by isolated setuptools 83.0.0**. This is not a claim that
the installed runtime dependency is patched or that the raw audit passes.
Do not use `--no-build-isolation` for distribution builds. Reassess this narrow
disposition when changing Torch, setuptools, or the build workflow; unrelated
advisories are not covered by it.
