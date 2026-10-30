# Bounded full-control experiment

Run these commands only with an explicitly authorized training and evaluation
budget. Start from the current repository checkout, not a historical backend
source directory. The backend source manifest must cover `environment.py`,
`multi_goal.py`, and `orientation.py`; a backend directory is never added to
the import path.

In WSL, select the checked-out repository as the current working directory and
set `PYTHON` to the intended Python interpreter. `PREVIOUS` is a closed,
previously qualified source run. All output paths below must be fresh absolute
paths. `DEPLOYMENT` must be on ext4 and outside the checkout; `MANIFEST` must
be outside the deployment. Create the separate V8 SDK from this checkout:

```sh
"$PYTHON" -B -m tools.record_rl1168.deployment "$PWD" "$DEPLOYMENT" "$MANIFEST"
export TMRL_V8_DEPLOYMENT="$DEPLOYMENT"
export TMRL_V8_MANIFEST="$MANIFEST"
export TMRL_V8_MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
```

Record the printed manifest SHA. Do not change or replace the deployment or the
manifest after qualification. In the same checkout and with those variables,
prepare and run the full-start qualification:

```sh
"$PYTHON" -B -m tools.record_rl1168.prepare "$PREVIOUS" "$PREPARATION" --seconds 60
"$PYTHON" -B -m tools.record_rl1168.qualify "$PREPARATION" "$QUALIFICATION" --trials 3
```

Both outputs have to pass. Preparation binds the plan, backend source manifest,
source pins and V8 manifest; qualification binds those same artifacts to the
measured full-start result. Any change requires fresh preparation and
qualification. Older unbound receipts are not accepted. Native qualification
uses real processes and its normal resource and closure guards; deployment
creation itself neither starts training nor establishes qualification.

From the same repository root in PowerShell, launch only after checking the
qualification result. Use absolute Windows paths for the three output/input
directories, absolute Linux paths for `LinuxPython`, `PrivateBase`,
`V8Deployment`, and `V8Manifest`, an explicit WSL distribution, and the agreed
UTC deadline:

```powershell
./tools/record_rl1168/launch.ps1 -Root $Root -Preparation $Preparation `
  -Qualification $Qualification -DeadlineUtc $DeadlineUtc -LinuxPython $LinuxPython `
  -PrivateBase $PrivateBase -WslDistro $WslDistro -V8Deployment $V8Deployment `
  -V8Manifest $V8Manifest -Comparison
```

Omit `-Comparison` for a single campaign. Each process checks the exact SDK
files against the current or frozen checkout and the qualified manifest before
importing `trackmaniarl`. The launcher's `source` directory contains a frozen
copy of the checkout SDK for this comparison. An incomplete or failed run is
not a qualifying result.
