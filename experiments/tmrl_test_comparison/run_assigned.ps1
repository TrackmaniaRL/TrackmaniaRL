param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('iqn', 'qr', 'sd-sac', 'dsac', 'discrete-sac', 'tqc', 'ppo', 'sac')]
    [string]$Algorithm,
    [switch]$Pilot,
    [ValidateSet(17, 29, 43)][int[]]$Seeds = @(17, 29, 43),
    [string]$ConfigDirectory
)
$ErrorActionPreference = 'Stop'
if ($Algorithm -in @('dsac', 'discrete-sac')) { $Algorithm = 'sd-sac' }
if ($Algorithm -eq 'sd-sac') {
    throw 'SD-SAC jest EXPERIMENTAL: wykluczony z pelnych treningow i standardowych pilotow. Patrz RUN_ALGORITHMS.md i link do archiwum; osobny projekt naprawczy wymaga jawnej zgody.'
}
if ($Pilot -and $ConfigDirectory) { throw 'Pilot nie moze uzywac katalogu dlugiej kampanii.' }
if (@($Seeds | Select-Object -Unique).Count -ne $Seeds.Count) { throw 'Seedy nie moga sie powtarzac.' }
$comparisonMutex = [System.Threading.Mutex]::new($false, 'Global\TrackmaniaRL.ComparisonController')
$comparisonMutexAcquired = $false
try {
    try { $comparisonMutexAcquired = $comparisonMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $comparisonMutexAcquired = $true }
    if (-not $comparisonMutexAcquired) {
        throw 'Inny launcher posiada blokade sterowania gra. Kolejka nie zostala uruchomiona.'
    }
$comparisonRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $comparisonRoot
$comparisonPython = Join-Path $comparisonRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $comparisonPython)) { throw 'Najpierw uruchom uv sync --group dev.' }
$comparisonStop = Join-Path $comparisonRoot "artifacts/STOP-$Algorithm"
$comparisonLegacyStops = if ($Algorithm -eq 'sd-sac') {
    @((Join-Path $comparisonRoot 'artifacts/STOP-discrete-sac'),
      (Join-Path $comparisonRoot 'artifacts/STOP-dsac'))
} else { @() }
function Test-ComparisonStop {
    if (Test-Path -LiteralPath (Join-Path $comparisonRoot 'artifacts/STOP')) { return $true }
    if (Test-Path -LiteralPath (Join-Path $comparisonRoot 'artifacts/tmrl-test-comparison/STOP')) { return $true }
    if (Test-Path -LiteralPath $comparisonStop) { return $true }
    foreach ($legacyStop in $comparisonLegacyStops) {
        if (Test-Path -LiteralPath $legacyStop) { return $true }
    }
    return $false
}
function Assert-ComparisonIdle {
    $comparisonBusy = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.CommandLine -match 'trackmaniarl\s+(train|resume|benchmark)\b' -or
            $_.CommandLine -match 'tmrl_test_comparison\.launch_checks\s+benchmark\b' -or
            $_.CommandLine -match 'tmrl_test_comparison\.preflight\b' -or
            $_.CommandLine -match '--comparison-map-preflight' -or
            $_.CommandLine -match 'runtime_helper\.py.*\bpreflight\b'
        }
    if ($comparisonBusy) { throw 'Inny trening lub ewaluacja steruje gra. Poczekaj na jego zakonczenie.' }
}
function Initialize-ComparisonMap([string]$ConfigPath) {
    $comparisonReceipt = Join-Path $comparisonRoot (
        'artifacts/tmrl-test-comparison/preflight/' + [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath) + '-' + [guid]::NewGuid().ToString('N') + '.json')
    $comparisonPreflightArgs = @('-m', 'experiments.tmrl_test_comparison.preflight', $ConfigPath,
        '--require-new-run', '--receipt', $comparisonReceipt, '--stop-file', $comparisonStop)
    foreach ($legacyStop in $comparisonLegacyStops) {
        $comparisonPreflightArgs += @('--stop-file', $legacyStop)
    }
    & $comparisonPython @comparisonPreflightArgs
    if ($LASTEXITCODE -ne 0) { throw 'Mapa nie jest gotowa. Trening nie zostal uruchomiony.' }
}
$comparisonSeeds = if ($Pilot) { @(17) } else { $Seeds }
foreach ($comparisonSeed in $comparisonSeeds) {
    if (Test-ComparisonStop) { throw 'STOP jest aktywny. Kolejka nie uruchomi kolejnego etapu.' }
    Assert-ComparisonIdle
    $comparisonStage = if ($Pilot) { 'pilot' } else { 'full' }
    $comparisonConfig = "experiments/tmrl_test_comparison/configs/$comparisonStage/$Algorithm-s$comparisonSeed.yaml"
    if ($ConfigDirectory) {
        $comparisonConfig = Join-Path $ConfigDirectory "$Algorithm-s$comparisonSeed.yaml"
    }
    if ($ConfigDirectory) {
        & $comparisonPython -m experiments.tmrl_test_comparison.campaign config-check $comparisonConfig
        if ($LASTEXITCODE -ne 0) { throw 'Konfiguracja kampanii jest niezgodna. Nie uruchomiono gry.' }
    }
    if (Test-ComparisonStop) { break }
    Initialize-ComparisonMap $comparisonConfig
    if (Test-ComparisonStop) { break }
    Assert-ComparisonIdle
    $comparisonTrainArgs = @('-m', 'trackmaniarl', 'train', $comparisonConfig, '--stop-file', $comparisonStop)
    & $comparisonPython @comparisonTrainArgs
    if ($LASTEXITCODE -ne 0) { throw "Trening $Algorithm seed $comparisonSeed nie powiodl sie. Kolejka zatrzymana." }
    if (Test-ComparisonStop) { break }
    if (-not $Pilot) {
        $comparisonCheckpoint = & $comparisonPython -m experiments.tmrl_test_comparison.launch_checks checkpoint $comparisonConfig
        if ($LASTEXITCODE -ne 0) { throw 'Brak potwierdzonego koncowego checkpointu. Wznow trening jawnie; kolejka zatrzymana.' }
        $comparisonCheckpoint = ($comparisonCheckpoint | Out-String).Trim()
        if (Test-ComparisonStop) { break }
        if ($Algorithm -eq 'ppo') {
            # PPO's trainer already evaluated exactly 30 trials after the final save.
            # Validate that measurement instead of silently collecting another 30.
            & $comparisonPython -m experiments.tmrl_test_comparison.launch_checks evaluation $comparisonConfig $comparisonCheckpoint
        } else {
            Assert-ComparisonIdle
            if (Test-ComparisonStop) { break }
            & $comparisonPython -m experiments.tmrl_test_comparison.launch_checks benchmark $comparisonConfig $comparisonCheckpoint --stop-file $comparisonStop
        }
        if ($LASTEXITCODE -ne 0) { throw 'Ewaluacja nie jest kompletnym poprawnym pomiarem 30 prob. Kolejka zatrzymana; zachowano artefakty.' }
    }
}
} finally {
    if ($comparisonMutexAcquired) { $comparisonMutex.ReleaseMutex() }
    $comparisonMutex.Dispose()
}
