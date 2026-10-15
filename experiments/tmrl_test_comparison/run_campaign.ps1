param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('jakub', 'borys', 'kamil', 'kuba-p')][string]$Worker,
    [switch]$Launch,
    [string]$StartAt
)
$ErrorActionPreference = 'Stop'
$campaignRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $campaignRoot
$campaignPython = Join-Path $campaignRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $campaignPython)) { throw 'Najpierw zainstaluj przypiete srodowisko.' }
& $campaignPython -m experiments.tmrl_test_comparison.campaign verify
if ($LASTEXITCODE -ne 0) { throw 'Plan lub przypiete pliki nie sa zgodne.' }
$campaignArguments = @('-m', 'experiments.tmrl_test_comparison.campaign', 'jobs', '--worker', $Worker)
if ($StartAt) { $campaignArguments += @('--start-at', $StartAt) }
$campaignJobs = & $campaignPython @campaignArguments
if ($LASTEXITCODE -ne 0) { throw 'Nie mozna pominac niekompletnych lub nieocenionych prob.' }
$campaignJobs = @($campaignJobs | ConvertFrom-Json)
if (-not $Launch) {
    $campaignJobs | Select-Object algorithm, seed, run_id | Format-Table
    Write-Output 'Plan sprawdzony. Gra nie zostala uruchomiona. -Launch rozpoczyna trening.'
    exit 0
}
foreach ($campaignJob in $campaignJobs) {
    # The assigned launcher owns the controller mutex; the supervisor handles STOP/closure.
    & $campaignPython -m experiments.tmrl_test_comparison.campaign run --worker $Worker --run-id $campaignJob.run_id
    if ($LASTEXITCODE -ne 0) { throw 'Proba niekompletna; nie rozpoczeto kolejnego seeda.' }
}
