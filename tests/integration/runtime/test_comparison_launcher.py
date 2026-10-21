"""Launcher guards must stop before opening a live game controller."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.tmrl_test_comparison import launch_checks, preflight
from experiments.tmrl_test_comparison.generate import configuration
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.spec import RunSpec


def test_preflight_refuses_existing_run_before_creating_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "artifacts" / "existing"
    existing.mkdir(parents=True)
    spec = SimpleNamespace(artifacts_dir=tmp_path / "artifacts", run_id="existing")
    monkeypatch.setattr(RunSpec, "from_yaml", lambda _: spec)

    def forbidden_controller(**kwargs: object) -> None:
        raise AssertionError("Controller created before existing-run guard")

    monkeypatch.setattr(preflight, "GamepadController", forbidden_controller)
    with pytest.raises(RuntimeError, match="Run already exists"):
        preflight.run_preflight(tmp_path / "config.yaml", require_new_run=True)


def test_launcher_delegates_to_bounded_preflight_and_preserves_each_receipt() -> None:
    launcher = (
        Path(__file__).resolve().parents[3] / "experiments/tmrl_test_comparison/run_assigned.ps1"
    )
    text = launcher.read_text(encoding="utf-8")
    assert "'experiments.tmrl_test_comparison.preflight'" in text
    assert "'--require-new-run'" in text
    assert "'--receipt'" in text
    assert "[guid]::NewGuid()" in text
    assert "'--stop-file', $comparisonStop" in text
    assert "'--stop-file', $legacyStop" in text
    assert "tmrl_test_comparison\\.preflight\\b" in text
    assert "confirm_finish()" not in text


def test_comparison_launcher_rejects_unqualified_sd_sac_and_busy_game(tmp_path: Path) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows launcher requires PowerShell")
    launcher = (
        Path(__file__).resolve().parents[3] / "experiments/tmrl_test_comparison/run_assigned.ps1"
    )
    script = tmp_path / "guards.ps1"
    script.write_text(
        r"""
param([string]$Launcher, [string]$Python)
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $Launcher, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Launcher syntax is invalid' }
foreach ($algorithm in @('sd-sac', 'discrete-sac', 'dsac')) {
    $blocked = $false
    try { & $Launcher $algorithm } catch { $blocked = $_.Exception.Message -match 'pelnych|Pelne' }
    if (-not $blocked) { throw "Unqualified algorithm was not blocked: $algorithm" }
}
$guard = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Assert-ComparisonIdle'
}, $true)
. ([scriptblock]::Create($guard.Extent.Text))
function Get-CimInstance { param($ClassName, $Filter) @() }
Assert-ComparisonIdle
function Get-CimInstance { param($ClassName, $Filter)
    [pscustomobject]@{CommandLine = 'python.exe -m trackmaniarl train config.yaml'}
}
$blocked = $false
try { Assert-ComparisonIdle } catch { $blocked = $_.Exception.Message -match 'steruje gra' }
if (-not $blocked) { throw 'Active controller was not blocked' }
$blocked = $false
$text = Get-Content -Raw -LiteralPath $Launcher
$text = $text.Replace(
    '$comparisonRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent',
    '$comparisonRoot = Split-Path (Split-Path (Split-Path $Launcher -Parent) -Parent) -Parent')
$text = $text.Replace(
    '$comparisonPython = Join-Path $comparisonRoot ''.venv/Scripts/python.exe''',
    '$comparisonPython = $Python')
$text = $text.Replace('Global\TrackmaniaRL.ComparisonController',
                      "Local\TrackmaniaRL.LauncherGuards.$PID")
$text = $text.Replace('& $comparisonPython', 'Invoke-BlockedPython')
function Invoke-BlockedPython { throw 'Unexpected Python invocation' }
$code = [scriptblock]::Create($text)
try { & $code sac } catch { $blocked = $_.Exception.Message -match 'steruje gra' }
if (-not $blocked) { throw 'Qualified SAC must reach the busy-game guard before any controller' }
Write-Output 'Launcher guards passed without game input'
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            str(launcher),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Launcher guards passed" in result.stdout


def test_launcher_sequences_three_seeds_and_stops_without_live_python(tmp_path: Path) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None:
        pytest.skip("Windows launcher requires PowerShell")
    launcher = (
        Path(__file__).resolve().parents[3] / "experiments/tmrl_test_comparison/run_assigned.ps1"
    )
    wrapper = tmp_path / "sequence.ps1"
    wrapper.write_text(
        r"""
param([string]$Launcher, [string]$StopPath, [string]$Python)
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $Launcher, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Launcher syntax is invalid' }
$preflight = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Initialize-ComparisonMap'
}, $true)
$text = Get-Content -Raw -LiteralPath $Launcher
$text = $text.Replace($preflight.Extent.Text, @'
function Initialize-ComparisonMap([string]$ConfigPath) {
    [void]$global:comparisonMockCalls.Add("preflight $ConfigPath")
}
'@)
$text = $text.Replace(
    '$comparisonRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent',
    '$comparisonRoot = Split-Path (Split-Path (Split-Path $Launcher -Parent) -Parent) -Parent')
$text = $text.Replace(
    '$comparisonStop = Join-Path $comparisonRoot "artifacts/STOP-$Algorithm"',
    '$comparisonStop = $StopPath')
$text = $text.Replace(
    '$comparisonPython = Join-Path $comparisonRoot ''.venv/Scripts/python.exe''',
    '$comparisonPython = $Python')
$text = $text.Replace('Global\TrackmaniaRL.ComparisonController',
                      "Local\TrackmaniaRL.LauncherSequence.$PID")
$text = $text.Replace('& $comparisonPython', 'Invoke-MockPython')
$code = [scriptblock]::Create($text)
function Get-CimInstance { param($ClassName, $Filter) @() }
function Invoke-MockPython {
    $items = @($args)
    [void]$global:comparisonMockCalls.Add(($items -join ' '))
    $global:LASTEXITCODE = 0
    if ($items[2] -eq 'checkpoint') { Write-Output 'C:\fake\final.pt' }
    if ($global:comparisonMockMode -eq 'stop' -and $items[2] -eq 'train') {
        Set-Content -LiteralPath $StopPath -Value 'stop'
    }
    if ($global:comparisonMockMode -eq 'failure' -and $items[2] -eq 'benchmark') {
        $global:LASTEXITCODE = 1
    }
}
foreach ($mode in @('iqn', 'ppo', 'pilot', 'stop', 'failure', 'after-failure')) {
    if (Test-Path -LiteralPath $StopPath) { Remove-Item -LiteralPath $StopPath }
    $global:comparisonMockCalls = [System.Collections.Generic.List[string]]::new()
    $global:comparisonMockMode = $mode
    $failed = $false
    try {
        if ($mode -eq 'pilot') { & $code iqn -Pilot }
        elseif ($mode -eq 'ppo') { & $code ppo }
        else { & $code iqn }
    } catch { $failed = $true }
    if ($failed -ne ($mode -eq 'failure')) { throw "Unexpected failure state for $mode" }
    $calls = @($global:comparisonMockCalls | Where-Object { $_ -notmatch '^preflight ' })
    if ($mode -in @('iqn', 'ppo', 'after-failure')) {
        if ($calls.Count -ne 9) { throw "Expected 9 sequential actions for $mode" }
        $measurement = if ($mode -eq 'ppo') { 'evaluation' } else { 'benchmark' }
        $seeds = @(17, 29, 43)
        for ($i = 0; $i -lt 3; $i++) {
            $seed = $seeds[$i]
            if ($calls[$i * 3] -notmatch "trackmaniarl train .*s$seed.yaml" -or
                $calls[$i * 3 + 1] -notmatch "launch_checks checkpoint .*s$seed.yaml" -or
                $calls[$i * 3 + 2] -notmatch "launch_checks $measurement .*s$seed.yaml") {
                throw "Wrong train/checkpoint/measurement order for $mode seed $seed"
            }
        }
    } elseif ($mode -eq 'failure') {
        if ($calls.Count -ne 3) { throw 'Evaluation failure started another seed' }
    } elseif ($calls.Count -ne 1 -or $calls[0] -notmatch 'trackmaniarl train ') {
        throw "Pilot or STOP started an unexpected action: $mode"
    }
}
Write-Output 'Mock launch sequences passed; no Python or game controller executed'
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            str(launcher),
            str(tmp_path / "STOP"),
            sys.executable,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Mock launch sequences passed" in result.stdout


def test_launcher_mutex_blocks_other_process_before_preflight(tmp_path: Path) -> None:
    shell = shutil.which("powershell.exe")
    if shell is None or os.name != "nt":
        pytest.skip("Windows named mutex requires Windows")
    launcher = (
        Path(__file__).resolve().parents[3] / "experiments/tmrl_test_comparison/run_assigned.ps1"
    )
    mutex_name = rf"Local\TrackmaniaRL.LauncherTest.{os.getpid()}"
    source = launcher.read_text(encoding="utf-8").replace(
        r"Global\TrackmaniaRL.ComparisonController", mutex_name
    )
    # A broken lock must still never invoke Python or open a controller in this test.
    source = source.replace("& $comparisonPython", "Invoke-BlockedPython")
    source = source.replace(
        "$ErrorActionPreference = 'Stop'",
        "function Invoke-BlockedPython { throw 'Unexpected Python invocation' }\n"
        "$ErrorActionPreference = 'Stop'",
    )
    guarded = tmp_path / "guarded.ps1"
    guarded.write_text(source, encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.ReleaseMutex.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateMutexW(None, True, mutex_name)
    assert handle
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(guarded), "iqn"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)
    assert result.returncode != 0
    assert "Inny launcher posiada blokade" in result.stderr
    assert "Unexpected Python invocation" not in result.stderr


@pytest.fixture
def launch_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> launch_checks.LaunchRun:
    spec = RunSpec.model_validate(configuration("iqn", 17, "full"))
    directory = tmp_path / spec.run_id
    directory.mkdir()
    monkeypatch.setattr(launch_checks, "run_fingerprint", lambda *_: "test-fingerprint")
    return launch_checks.LaunchRun(tmp_path / "run.yaml", spec, directory)


def _saved_checkpoint(run: launch_checks.LaunchRun, family: str) -> tuple[Path, dict]:
    distributed = family == "distributed-update"
    updates = 509500 if distributed else 2048000 // run.spec.training.sequence_length
    checkpoint = run.directory / "checkpoints" / f"{family}-{updates:08d}.pt"
    state = {
        "run_fingerprint": "test-fingerprint",
        "learner": {"model": torch.tensor([1.0]), "processed_transitions": 2048000},
        "distributed" if distributed else "counters": {
            "transitions": 2048000,
            "updates": updates,
            "update_credit": 0.0,
            "fractional_updates": 0.0,
        },
    }
    TorchCheckpointCodec().save(state, checkpoint)
    events = [
        {"event": event, "payload": {"path": str(checkpoint)}}
        for event in ("train/checkpoint", "train/checkpoint_completed")
    ]
    (run.directory / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    return checkpoint, state


@pytest.mark.parametrize("fault", ["fractional_credit", "missing_update", "unprocessed"])
def test_final_checkpoint_refuses_collected_but_unlearned_ppo_rollout(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    ppo = RunSpec.model_validate(configuration("ppo", 17, "full"))
    run = launch_checks.LaunchRun(launch_run.config, ppo, launch_run.directory)
    checkpoint, state = _saved_checkpoint(run, "update")
    if fault == "fractional_credit":
        state["counters"]["fractional_updates"] = 1.0
    elif fault == "missing_update":
        # Use a matching saved filename so this exercises the rollout budget check.
        state["counters"]["updates"] = 999
        new_path = checkpoint.with_name("update-00000999.pt")
        events = (run.directory / "events.jsonl").read_text(encoding="utf-8")
        (run.directory / "events.jsonl").write_text(
            events.replace(checkpoint.name, new_path.name), encoding="utf-8"
        )
        checkpoint = new_path
    else:
        state["learner"]["processed_transitions"] = 2045952
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match="incomplete rollout"):
        launch_checks.final_checkpoint(run)


@pytest.mark.parametrize("family", ["distributed-update", "update"])
def test_final_checkpoint_accepts_completed_async_and_ppo(
    launch_run: launch_checks.LaunchRun, family: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, family)
    assert launch_checks.final_checkpoint(launch_run) == checkpoint


@pytest.mark.parametrize(
    "change",
    [
        ("transitions", 145408, "completed training budget"),
        ("updates", 500000, "completed training budget"),
        ("update_credit", 1.0, "undrained"),
        ("update_credit", float("nan"), "undrained"),
    ],
)
def test_final_checkpoint_refuses_incomplete_training(
    launch_run: launch_checks.LaunchRun, change: tuple[str, float, str]
) -> None:
    field, value, error = change
    checkpoint, state = _saved_checkpoint(launch_run, "distributed-update")
    state["distributed"][field] = value
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match=error):
        launch_checks.final_checkpoint(launch_run)


def test_final_checkpoint_does_not_fall_back_to_older_success(
    launch_run: launch_checks.LaunchRun,
) -> None:
    _saved_checkpoint(launch_run, "distributed-update")
    pending = launch_run.directory / "checkpoints/distributed-update-00509501.pt"
    with (launch_run.directory / "events.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            "\n" + json.dumps({"event": "train/checkpoint", "payload": {"path": str(pending)}})
        )
    with pytest.raises(RuntimeError, match="not successfully saved"):
        launch_checks.final_checkpoint(launch_run)


@pytest.mark.parametrize("fault", ["fingerprint", "nonfinite"])
def test_final_checkpoint_refuses_changed_code_or_broken_weights(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    checkpoint, state = _saved_checkpoint(launch_run, "update")
    if fault == "fingerprint":
        state["run_fingerprint"] = "different"
    else:
        state["learner"]["model"] = torch.tensor([float("nan")])
    TorchCheckpointCodec().save(state, checkpoint)
    with pytest.raises(RuntimeError, match=r"fingerprint|non-finite"):
        launch_checks.final_checkpoint(launch_run)


def _measurement(run: launch_checks.LaunchRun, checkpoint: Path) -> tuple[Path, dict]:
    assert run.spec.evaluation is not None
    map_spec = run.spec.evaluation.maps[0]
    payload = {
        "status": "complete",
        "expected_trials": 30,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metrics": {"eval/finish_time_s": 55.0, "eval/median_finish_time_s": 55.0},
        "trials": [
            {
                "map_id": map_spec.id,
                "map_uid": map_spec.expected_map_uid,
                "trial_index": index,
                "finished": True,
                "steps": 100,
                "step_race_time_ms_max": 50,
                "step_race_time_measurement_count": 100,
                "step_race_time_measurements_valid": True,
                "telemetry_error": None,
                "controller_error": None,
                "termination_reason": "finished",
            }
            for index in range(30)
        ],
    }
    path = run.directory / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_complete_slow_measurement_is_reported_as_failed_gate_without_repeating(
    launch_run: launch_checks.LaunchRun, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "update")
    path, _ = _measurement(launch_run, checkpoint)
    launch_checks.report_measurement(launch_run, checkpoint, path)
    assert "configured benchmark gate failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault", ["cancelled", "missing_trial", "duplicate", "wrong_hash", "controller", "timing"]
)
def test_measurement_refuses_incomplete_or_invalid_results(
    launch_run: launch_checks.LaunchRun, fault: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "update")
    path, payload = _measurement(launch_run, checkpoint)
    if fault == "cancelled":
        payload["status"] = "cancelled"
    elif fault == "missing_trial":
        payload["trials"].pop()
    elif fault == "duplicate":
        payload["trials"][1]["trial_index"] = 0
    elif fault == "wrong_hash":
        payload["checkpoint_sha256"] = "wrong"
    elif fault == "controller":
        payload["trials"][0]["controller_error"] = "failed"
    else:
        payload["trials"][0]["step_race_time_ms_max"] = 140
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        launch_checks.measurement_gate(launch_run, checkpoint, path)


@pytest.mark.parametrize("failure", ["performance", "unrelated"])
def test_benchmark_only_continues_after_exact_performance_failure(
    launch_run: launch_checks.LaunchRun, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    checkpoint, _ = _saved_checkpoint(launch_run, "distributed-update")
    source, _ = _measurement(launch_run, checkpoint)
    target = launch_run.directory.with_name(launch_run.spec.run_id + "-benchmark-test")

    def fake_benchmark(_args: object) -> None:
        target.mkdir()
        artifact = target / "evaluation.json"
        shutil.copyfile(source, artifact)
        gate = launch_checks.measurement_gate(launch_run, checkpoint, artifact)
        error = launch_checks._benchmark_gate_failure(gate)
        raise RuntimeError(error if failure == "performance" else "Unrelated failure")

    monkeypatch.setattr(launch_checks, "_benchmark", fake_benchmark)
    stop = launch_run.directory / "STOP"
    if failure == "performance":
        launch_checks.benchmark_final(launch_run, checkpoint, stop)
    else:
        with pytest.raises(RuntimeError, match="Unrelated failure"):
            launch_checks.benchmark_final(launch_run, checkpoint, stop)
