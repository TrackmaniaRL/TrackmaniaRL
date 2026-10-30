param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root)
foreach($name in @('launch-intent.json','windows-launch.json','pipeline-state.json','pipeline-launch-marker')){
 if(Test-Path -LiteralPath "$Root/$name"){throw 'Existing Fall launch must be inspected; no duplicate launch'}
}
$admission=Get-Content -LiteralPath "$Root/environment-admission.json" -Raw|ConvertFrom-Json
if($admission.status -ne 'PASS_FALL05_CAUSAL_PHYSICS_STEP_ENVIRONMENT'){throw 'Actual environment admission required'}
if(-not(Test-Path -LiteralPath "$Root/launch-source-manifest.json")){throw 'Sealed launch sources required'}
$linuxRoot='/mnt/f/'+$Root.Substring(3).Replace('\','/')
$private='/home/kuba/tmrl-full-qr-20261022-011/'+(Split-Path -Leaf $Root)
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
$intent=[IO.File]::Open("$Root/launch-intent.json",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);$intent.Dispose()
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--cd',"$private/source",'--',$python,'-B',"$private/source/pipeline.py",$linuxRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/windows-parent.stdout" -RedirectStandardError "$Root/windows-parent.stderr"
$record=[ordered]@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;private=$private;queued=$true;gameStarted=$false}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$guard=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$guard.Id;birthUtcTicks=$guard.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
