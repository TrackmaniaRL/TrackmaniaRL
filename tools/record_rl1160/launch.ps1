param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root)
foreach($name in @('launch-intent.json','windows-launch.json','state.json','launch-marker')){
 if(Test-Path -LiteralPath "$Root/$name"){throw 'An existing launch must be inspected; never launch a duplicate'}
}
$audit=Get-Content -LiteralPath "$Root/offline-audit.json" -Raw|ConvertFrom-Json
if($audit.status -ne 'PASS_OFFLINE_REAL_DATA_AUDIT'){throw 'CPU real-observation audit required'}
$name=Split-Path -Leaf $Root
$linuxRoot='/mnt/f/'+$Root.Substring(3).Replace('\','/')
$private="/home/kuba/tmrl-full-qr-20261022-011/$name"
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
$dependency=@()
foreach($p in Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'"){
 if($p.CommandLine -like '*run-qualification-jguard-20261029.py*qualification-4h-0acad837f044470d9e8801c19fbec488*'){
  $live=Get-Process -Id $p.ProcessId
  $dependency+=@{pid=$live.Id;birthUtcTicks=$live.StartTime.ToUniversalTime().Ticks.ToString()}
 }
}
if($dependency.Count -eq 0){throw 'Known dependency owner disappeared; inspect its completion before launch'}
[IO.File]::WriteAllText("$Root/windows-dependency.json",(ConvertTo-Json -InputObject @($dependency)))
$intent=[IO.File]::Open("$Root/launch-intent.json",[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
$intent.Dispose()
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--cd',"$private/source",'--',$python,'-B',"$private/pipeline.py",$linuxRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/windows-parent.stdout" -RedirectStandardError "$Root/windows-parent.stderr"
$record=[ordered]@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;linuxRoot=$linuxRoot;private=$private;queued=$true;gameStarted=$false}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$guard=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$guard.Id;birthUtcTicks=$guard.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
