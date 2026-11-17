# Roadmap

These items describe proposed follow-up work; they are not release commitments.

## Native GPU validation

Validate the CUDA 13.2 development environment on the target RTX GPU with
native collection, checkpoint restore and a controlled stop. Keep the software
test results separate from open questions about analog physics parity and
end-to-end training performance.

## Distributed evaluation across multiple maps

Scheduled distributed evaluation currently accepts only one attested map. Extend
it to run complete suites across maps when the backend can safely load and verify
each map. The design must bind every trial to the assigned policy, map and assets,
rebuild or isolate map-dependent pipelines, and restore the training map before
collection resumes. Backends that cannot verify a map switch should fail explicitly.

Validate interrupted runs and retries, complete trial accounting, map identity
and asset mismatches, and restoration of the training environment. Exercise the
implemented backend with real maps before claiming multi-map support.

## Journal and replay validation

The journal and replay refactor defines ownership of stored data and separates
evaluation operations from the journal core. Validate these boundaries on longer
native collection workloads and measure end-to-end impact before claiming a
performance improvement.

Preserve checkpoint and journal compatibility where supported, with migration
coverage and tests for concurrent append, sampling, snapshots and recovery.
