# Roadmap

These items describe proposed follow-up work; they are not release commitments.

## Distributed evaluation across multiple maps

Scheduled distributed evaluation includes an experimental, explicitly enabled
OpenPlanet multi-map path; the default remains an attested single map. The
experimental path requires both `enable_experimental_multimap: true` and
`assume_exclusive_no_unobserved_map_changes: true`. It assumes exclusive control
of a trusted instance without autonomous or unobserved map changes; the checks
do not prove continuous map identity. The plugin and backend require real-game
validation before support can be claimed. The design binds assigned policy and
asset identity, checks observed map identity, isolates map-dependent pipelines,
and restores the training map before collection resumes. An observed mismatch
fails closed.

Validate interrupted runs and retries, complete trial accounting, map identity
and asset mismatches, and restoration of the training environment. Exercise the
implemented backend with real maps before claiming multi-map support.

## Journal and replay refactoring

In a separate change, consider splitting journal responsibilities and simplifying
the sequence sampler. Clarify ownership and copying of replay snapshots so that
concurrency guarantees are visible at component boundaries.

Preserve checkpoint and journal compatibility where supported, with migration
coverage and tests for concurrent append, sampling, snapshots and recovery. Measure
end-to-end impact before claiming performance improvements.
