> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Literal Windows consumer controls replayed through native V7

On 21 October 2026, the two retained 007 Windows captures reproduced every
observed next physical state through the existing pinned native V7 engine when
each row's actual gas/brake/steer words were applied to the following row's pose.
The short capture matched 198/198 targets and the long mild-left capture matched
227/227. Every matched target was identical in all ten float32 words:
quaternion4, position3 and velocity3. Vertical position and velocity differences
were zero. The two deliberately shifted phase hypotheses were also executed and
retained; both diverged.

This is a diagnostic replay of already observed physical-consumer state. It does
not establish an acknowledgement for a Windows command issued by the controller,
a Windows race-clock mapping, an atomic cross-probe join, or paired training.

## Inputs and the predeclared phase experiment

The implementation is `scripts/input_consumer_native_replay.py`. Its inputs are
an explicitly selected `input-consumer007.json` and the retained actual neutral
native POST sequence `neutral-clock001/report.json`. It validates the 007 capture
and exact 64-byte restoration before using its records. All raw R13D increments
must equal ten; this is a capture consistency requirement and does not itself
qualify that argument as a clock. No Windows race timestamp participates in
choosing the native anchor.

The first captured physical position+velocity, six literal float32 words, must
have exactly one content match in the retained neutral native POSTs. Ambiguous
or absent matches fail. The observed anchors were nativeT2250 for live002 and
nativeT2260 for live003. Every case independently starts at literal2190, seeks
with neutral inputs, and requires the fresh full ten-word pose to equal the
retained native state at its declared start. The six-word content match selects
the anchor; the full ten-word check then verifies the fresh native seek.

For N rows, the following hypotheses are fixed before inspecting outcomes:

| Case | Native start | Source control → target pose | Compared targets |
| --- | --- | --- | --- |
| A, current-to-next | content anchor, row0 pose | row i → row i+1, i=0..N−2 | N−1 |
| B, previous-to-next | anchor+10, row1 pose | row i−1 → row i+1, i=1..N−2 | N−2 |
| C, current-to-same | retained neutral anchor−10 | row i → row i, i=0..N−1 | N |

A retains the last control without applying it: there is no observed following
pose for that control. B starts at the actually matched row1 state, avoiding an
invented control preceding row0. C runs only when the preceding neutral POST
was actually retained. All three results remain in the report; the script does
not choose a winning offset or replace a failing phase.

The actual 007 words are `[gas, brake, steer, filtered steer]`. Only the first
three are reordered to native `[steer, gas, brake]` without decimal conversion,
rounding or canonical action IDs. In particular, the measured mild-left word
`0xbe2952a5` is preserved exactly. Filtered steer is observed evidence, not an
independently written native input. Each transition retains the existing V7
current-record receipt, write audit and native POST, including proof that the
native actual gas/brake/steer bytes equal the supplied literal words.

## Retained results

Counts below are exact for both physical6 and physical10. Deltas are native
minus the corresponding Windows target; no tolerance is used for the counts.

| Capture | Case | Exact / targets | Endpoint Y difference | Endpoint VY difference |
| --- | --- | --- | --- | --- |
| live002, 199 rows | A | 198/198 | 0 | 0 |
| live002 | B | 49/197 | +0.00008487701416015625 | −0.0014237016439437866 |
| live002 | C | 50/199 | −0.00006771087646484375 | +0.001536056399345398 |
| live003, 228 rows | A | 227/227 | 0 | 0 |
| live003 | B | 13/226 | +0.000240325927734375 | +0.00125519884750247 |
| live003 | C | 14/228 | +0.00007343292236328125 | −0.0017589381895959377 |

The complete three-case runs took 11.916906245 seconds for live002 and
12.731187957 seconds for live003. All six independently supervised sessions
closed with `remainingOwned=[]`. The corresponding Windows captures had
already verified explicit stop, exact original-code restoration, input-profile
restoration and controller release.

Evidence remains outside the repository:

- `<USER_HOME>/tmrl-headless-parity-20261021-005/consumer-native-replay002/report.json`,
  SHA256 `d2b1e417ed1903081696a8a02035ef9186eef92ce5b5686ad79b18acb0eb37b0`.
- `<USER_HOME>/tmrl-headless-parity-20261021-005/consumer-native-replay003/report.json`,
  SHA256 `cc42de57f90037cb7da54c608e3959670960885d88cab7469c81752026ad3e3c`.
- The reports retain source-file hashes, explicit row mappings, every compared
  pose and native receipt/audit/POST, first mismatch and endpoint, administrative
  neutral seeks, and independent cleanup records.

The live003 replay was executed before a presentation-only Ruff reformat of the
new Python source; live002 used the formatted source. Both precede the reviewed
failure-path improvement that retains cleanup diagnostics even if `close()`
raises. The reports hash their input artifacts but do not attest the exact
Python source at execution. The recorded literal native receipts and observations
remain the evidence for the numeric results.

## Repeating the bounded experiment

Run inside WSL from the shared worktree, selecting an existing retained capture
and a fresh absolute ext4 output path. The helper provisions only approved pinned
asset/support files into that output's new `deployment` directory. It does not
copy prior results or alter the historical001 runtime, source/build pins or
production reset profile.

```bash
cd /mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  <USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python \
  -m scripts.input_consumer_native_replay \
  --trace /mnt/f/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-live003/input-consumer007.json \
  --neutral-report <USER_HOME>/tmrl-headless-parity-20261021-005/neutral-clock001/report.json \
  --output <USER_HOME>/tmrl-headless-parity-20261021-005/consumer-native-replay-NEW
```

Change only `--trace` and the fresh `--output` to replay live002. `--v6-config`
and `--receipt` explicitly select alternative approved source locations if
needed. The output must not exist, must be canonical and outside `/mnt`, and
must have no symlink parents. The total experiment has a 170-second cooperative
deadline and observes an output-directory `STOP` file between native requests.
Every case closes in `finally`; missing cleanup proof or any owned survivors
fails. It does not start Windows Python, control a game, change a live plugin,
or run training.

## Validation and remaining scope

Ten pure tests in
`tests/unit/trackmania/test_input_consumer_native_replay.py` passed. They cover
unique content anchors, explicit phase indices/counts, absence of an invented
endpoint, exact detection of a one-ULP vertical difference, independent
quaternion comparison, strict uint32 words, adjacent steering-bit retention,
cleanup diagnostic retention on a real close exception, and rejection of missing
or nonempty zero-owned cleanup proof.
Ruff check/format passed for both new files. The final project audit reports
typing checks separately. The six real native sessions above provide the
engine-level evidence; the pure tests do not simulate that evidence.

These results explain why using the real consumer sequence removes the observed
vertical replay discrepancy for these two traces. They do not establish that
every future Windows command is applied once at a requested tick. A reusable
training interface still needs an independently qualified command-to-consumer
join, its actual duration contract and missing/terminal/failure handling.
