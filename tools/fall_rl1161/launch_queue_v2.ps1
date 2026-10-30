param([Parameter(Mandatory=$true)][string]$Root,[Parameter(Mandatory=$true)][string]$Private,[string]$QueueName='queue-v2')
$ErrorActionPreference='Stop'
$queue=Join-Path $Root $QueueName
if(Test-Path -LiteralPath "$queue/windows-launch.json"){throw 'No duplicate queue-v2 launch'}
$old=Get-Content -LiteralPath "$Root/windows-launch.json" -Raw|ConvertFrom-Json
$oldGuard=Get-Content -LiteralPath "$Root/power-guard-launch.json" -Raw|ConvertFrom-Json
foreach($record in @($old,$oldGuard)){
 $p=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
 if($p -and $p.StartTime.ToUniversalTime().Ticks.ToString() -eq $record.birthUtcTicks){throw 'Original exact owner/guard still active'}
}
$guard=Get-Content -LiteralPath "$Root/keep_awake.ps1" -Raw
[IO.File]::WriteAllText("$queue/keep_awake.ps1",$guard)
$linuxRoot='/mnt/f/'+$Root.Substring(3).Replace('\','/')
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
$intent=[IO.File]::Open("$queue/launch-intent",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);$intent.Dispose()
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--cd',"$Private/source",'--',$python,'-B',"$Private/$QueueName/pipeline.py",$linuxRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$queue/windows-parent.stdout" -RedirectStandardError "$queue/windows-parent.stderr"
$record=@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;private=$Private}
[IO.File]::WriteAllText("$queue/windows-launch.json",($record|ConvertTo-Json))
$power=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$queue/keep_awake.ps1",'-Root',$queue) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$queue/power-guard.stdout" -RedirectStandardError "$queue/power-guard.stderr"
[IO.File]::WriteAllText("$queue/power-guard-launch.json",(@{pid=$power.Id;birthUtcTicks=$power.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
