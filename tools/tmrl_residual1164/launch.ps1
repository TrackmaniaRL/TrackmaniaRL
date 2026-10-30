param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root)
if(Test-Path -LiteralPath "$Root/windows-launch.json"){throw 'No duplicate direct record RL launch'}
$plan=Get-Content -LiteralPath "$Root/plan.json" -Raw|ConvertFrom-Json
foreach($p in Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'"){
 if($p.CommandLine -like '*campaign_entry*--role*learner*'){throw 'Inspect active learner before launch'}
}
$guard=Get-Content tools/native_line1134/keep_awake.ps1 -Raw
$guard=$guard.Replace('Owned bounded headless native line search; release with its exact Windows owner','Owned tmrl neural residual PPO experiment; release with exact owner').Replace('AddHours(10).AddMinutes(10)','AddHours(7)')
[IO.File]::WriteAllText("$Root/keep_awake.ps1",$guard)
$linuxRoot='/mnt/f/'+$Root.Substring(3).Replace('\','/')
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
$intent=[IO.File]::Open("$Root/launch-intent",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);$intent.Dispose()
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--cd',"$($plan.source)",'--',$python,'-B',"$($plan.source)/run.py",$linuxRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/windows-parent.stdout" -RedirectStandardError "$Root/windows-parent.stderr"
$record=@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startedUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;private=$plan.private}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$power=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$power.Id;birthUtcTicks=$power.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
