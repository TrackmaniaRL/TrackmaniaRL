# Episode metric provenance

Distributed `train/episode` and `eval/episode` records include `phase`,
`metric_segment_id`, `actor_session_id`, `actor_chunk_sequence`, and
`accepted_transitions_at_ingest`. The existing `policy_version` belongs to the
episode summary; it is not inferred from the coordinator's current version.

`metric_segment_id` is new for each coordinator instance, including resumes.
The actor session and chunk sequence identify the producing stream. The
transition count is the coordinator's exact accepted count **after the whole
submitted chunk is applied**. It is not the count when that episode ended:
chunks can contain multiple episodes, and different actors may be ingested in
a different order from their collection times. Evaluation does not itself add
training transitions. Do not use these values as exact episode-end positions.

During WAL recovery already applied episode events are not re-emitted, to
avoid silently duplicating metric records. Preserve prior segment logs and
mark any gaps rather than interpreting missing events as failed episodes.
For reporting, compute finish rate over all attempts and compute time
statistics only over completed attempts, separately for training and evaluation.
