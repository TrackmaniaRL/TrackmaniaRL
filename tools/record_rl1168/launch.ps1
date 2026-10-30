param([Parameter(Mandatory=$true)][string]$Root,
      [Parameter(Mandatory=$true)][string]$Preparation,
      [Parameter(Mandatory=$true)][string]$Qualification,
      [Parameter(Mandatory=$true)][string]$DeadlineUtc,
      [switch]$Comparison)
$ErrorActionPreference='Stop'
$repo='C:/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania'
if((Get-Location).Path.Replace('\','/') -ne $repo){throw 'Explicit approved workdir required'}
if(!$DeadlineUtc.EndsWith('Z')){throw 'New absolute UTC deadline ending in Z required'}
$end=[DateTimeOffset]::Parse($DeadlineUtc)
$budget=($end-[DateTimeOffset]::UtcNow).TotalSeconds
if($budget -lt 1860 -or $budget -gt 36000){throw 'New budget must be31min..10h'}
$Root=[IO.Path]::GetFullPath($Root)
if(!$Root.StartsWith('F:\AITrackmania\reports\') -or (Test-Path -LiteralPath $Root)){throw 'Fresh F reports root required'}
$proof=Get-Content -LiteralPath "$Qualification/result.json" -Raw|ConvertFrom-Json
$receipt=Get-Content -LiteralPath "$Preparation/preparation.json" -Raw|ConvertFrom-Json
if($proof.status -ne 'PASS_FULL_START_BC' -or $proof.cleanupErrors.Count -ne 0 -or $proof.trials.Count -ne 3 -or $proof.checkpoint_sha256 -ne $receipt.checkpoint_sha256){throw 'Qualified initializer required'}
$running=Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'wsl.exe'"
if($running|Where-Object {$_.CommandLine -match 'record_rl1168.(supervise|train|compare)|vision_qr1167.run|vision_qr1167.*entry.py|tmrl_residual1164.*run.py|campaign_entry.*--role.*learner'}){throw 'Inspect existing campaign before new launch'}
New-Item -ItemType Directory -Path "$Root/source/tools/record_rl1168","$Root/source/tools/vision_qr1167" -Force|Out-Null
Copy-Item -LiteralPath (Get-ChildItem -LiteralPath "$repo/tools/record_rl1168" -Filter '*.py').FullName -Destination "$Root/source/tools/record_rl1168"
Copy-Item -LiteralPath "$repo/tools/vision_qr1167/returns.py" -Destination "$Root/source/tools/vision_qr1167"
[IO.File]::WriteAllText("$Root/source/tools/__init__.py",'')
[IO.File]::WriteAllText("$Root/source/tools/vision_qr1167/__init__.py",'')
$hashes=@{}
Get-ChildItem -LiteralPath "$Root/source" -Recurse -File|ForEach-Object {
 $relative=$_.FullName.Substring(("$Root/source").Length+1).Replace('\','/')
 $hashes[$relative]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
}
[IO.File]::WriteAllText("$Root/source-sha256.json",($hashes|ConvertTo-Json))
function Linux-Path([string]$path){
 $absolute=[IO.Path]::GetFullPath($path)
 if(!$absolute.StartsWith('F:\AITrackmania\reports\')){throw 'Expected F report path'}
 return '/mnt/f/'+$absolute.Substring(3).Replace('\','/')
}
$linuxRoot=Linux-Path $Root
$launch=@{preparation=(Linux-Path $Preparation);qualification=(Linux-Path $Qualification);deadline_utc=$DeadlineUtc}
[IO.File]::WriteAllText("$Root/launch.json",($launch|ConvertTo-Json))
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
$module=if($Comparison){'tools.record_rl1168.compare'}else{'tools.record_rl1168.supervise'}
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--cd',"$linuxRoot/source",'--',$python,'-B','-m',$module,$linuxRoot) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/parent.stdout" -RedirectStandardError "$Root/parent.stderr"
$record=@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startedUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;deadlineUtc=$DeadlineUtc}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$guard=Get-Content -LiteralPath "$repo/tools/native_line1134/keep_awake.ps1" -Raw
$guard=$guard.Replace('Owned bounded headless native line search; release with its exact Windows owner','Owned bounded full-control QR; release with exact Windows owner')
[IO.File]::WriteAllText("$Root/keep_awake.ps1",$guard)
$power=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$power.Id;birthUtcTicks=$power.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
