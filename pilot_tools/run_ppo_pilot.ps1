param([switch]$Launch)
$ErrorActionPreference = 'Stop'
$pilotRoot = Split-Path $PSScriptRoot -Parent
$pilotPython = Join-Path $pilotRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pilotPython)) {
    $pilotPython = 'H:/Studia/inzynierskie/inzynierkav2/AITrackmania/.venv/Scripts/python.exe'
}
Set-Location -LiteralPath $pilotRoot
if (-not (Test-Path -LiteralPath $pilotPython)) { throw 'Pinned Python environment missing.' }
$pilotCommand = if ($Launch) { 'launch' } else { 'verify' }
& $pilotPython -m pilot_tools.ppo_pilot $pilotCommand
if ($LASTEXITCODE -ne 0) { throw 'Pilot verification/launch failed. Inspect evidence; no retry.' }
