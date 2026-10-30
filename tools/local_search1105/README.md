# Local control search 1105 — prepared, NOT started

This is a bounded search for a better **control sequence**, not an autonomous
policy evaluation and not a public WR claim. Run 1096 and its deployed sources
are not modified. No simulator was started during preparation.

## Search space

The qualified Jimmy 17,715 ms plan has 355 five-tick programs. The first 1771
native 10 ms inputs match the digital CSV exactly with the already qualified
source offset +1. The remaining four neutral padding ticks are protected.
Search never changes countdown, reset alignment, map, physics or rewards.

Each candidate changes only one contiguous interval of one or two ticks:

* advance or delay an existing control boundary by 10/20 ms;
* move steering by one native bin (1/6), preserving gas and brake;
* toggle gas or brake for 10/20 ms, preserving the other channels.

No new analog interpolation, guessed telemetry or artificial Goal is used.
Canonical motor IDs exclude pulse aliases; programs use individual tick entries.
All edits are deduplicated. Seed 1105 interleaves five mutation families over
the entire lap. The first batch contains 512 of 11,894 possible single edits.
Its separate 439-program catalog retains the original 132 programs. It does
not change the action head or vocabulary of run 1096.

This first experiment evaluates independent mutations of Jimmy. It does not
compound changes greedily or search every combination. A later, separately
approved round could use a confirmed faster sequence as its new incumbent.

## Offline preparation

`python -B -m tools.local_search1105.prepare --study <ROOT> --budget 512`

This only creates `ROOT/local-search1105`, refuses overwrites, verifies baseline
inputs and hashes sources. It imports neither the SDK nor the native runtime.
The output has `plan.json`, `candidate-plans.npz`, and `macro-catalog.npz`.
Row zero is the unmodified baseline. Every candidate is linked to its edit,
row index, exact motor-sequence SHA and input provenance.

## Future execution — NOT authorized or performed in this preparation

Prepared entry point: `tools.local_search1105.run`. Its explicit
`--execute-after-1096` flag is deliberately absent from preparation. Execution
requires Linux, completion of 1096, successful final checkpoint audit, all four
worker exits zero, empty cleanup lists, exact old process identities gone and
the old power request released. A missing proof aborts, rather than stopping
1096. The execution directory is exclusive; no duplicate launch or resume.

Before launch, use the pinned renderer Python and this isolated checkout via
WSL, review the generated artifacts and run static tests again. The native
adapter has only static checks so far: no live qualification is claimed.

The future runner is CPU-only, one native trial at a time, with fresh seed-zero
resets. It uses the existing qualified factory, native Goal accessor and owned
pidfd cohort. Each trial retains config, launch identities, native archive,
receipt and closure. Native guards remain 50 ms/150 s/Goal/material/contact.
DNF is preserved with a null finish time; backend failure aborts the search.
No reward ranking, physical endpoint or nearest-line proxy selects winners.

1. Qualify the unchanged sequence five times at exactly native Goal 17715 ms.
   Otherwise stop; do not tune clock offsets to force a match.
2. Evaluate the prepared candidates in reproducible order.
3. For any strict improvement over the best confirmed time, repeat four more
   times. Accept only five identical actual Goal times, all faster.
4. Save the exact winning row and archive evidence. A one-off fast trial that
   does not repeat stays in history but does not replace the incumbent.

The run has a conservative one-hour total budget including cleanup, 240 s
supervision per trial, and reserves 1200 s before a new candidate to allow its
five-trial confirmation. Thus the full 512 candidates may not fit. Actual trial
count and budget stop are reported, never presented as exhaustive search.
Storage reserves: C 10 GiB, Linux/F 4 GiB, owned search quota 8 GiB.
Only `execution/STOP` and a trial's own STOP are used. Nothing auto-starts.

## What follows a faster sequence

Successful search is a new demonstration, not proof the neural agent can drive
it. Collect only actual native observations/actions/rewards from that sequence,
then separately authorize training and greedy evaluation with teacher files
denied. New programs require a separately versioned model/action mapping.
Any public-record claim additionally needs independent native-client validation.

## Checks

Pure tests cover locality/padding, exact edge timing, deduplication, selection,
canonical controls and repeatability acceptance. Ruff and strict mypy check all
five files. Unit tests do not run physics or train a model.
