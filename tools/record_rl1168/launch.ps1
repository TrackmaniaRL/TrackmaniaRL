<#
.SYNOPSIS
Start a fresh bounded campaign from the current repository root.
.DESCRIPTION
Root, Preparation and Qualification are absolute Windows drive paths accessible
to the named WSL distribution. LinuxPython and PrivateBase are absolute Linux
paths to an existing executable and an existing writable directory. DeadlineUtc
is an absolute UTC timestamp ending in Z. Comparison runs the retention ablation.
V8Deployment and V8Manifest are canonical Linux paths to a fresh SDK deployment
created with `python -B -m tools.record_rl1168.deployment SOURCE DEPLOYMENT MANIFEST`.
The qualification must have been run against that exact deployment.
The caller is responsible for authorizing the deadline and avoiding overlap.
#>
param([Parameter(Mandatory=$true)][string]$Root,
      [Parameter(Mandatory=$true)][string]$Preparation,
      [Parameter(Mandatory=$true)][string]$Qualification,
      [Parameter(Mandatory=$true)][string]$DeadlineUtc,
      [Parameter(Mandatory=$true)][string]$LinuxPython,
      [Parameter(Mandatory=$true)][string]$PrivateBase,
      [Parameter(Mandatory=$true)][string]$WslDistro,
      [Parameter(Mandatory=$true)][string]$V8Deployment,
      [Parameter(Mandatory=$true)][string]$V8Manifest,
      [switch]$Comparison)
$ErrorActionPreference='Stop'
$repo=(& git rev-parse --show-toplevel)
if($LASTEXITCODE -ne 0 -or !$repo){throw 'Current directory must be a repository root'}
$repo=[IO.Path]::GetFullPath($repo)
if(![string]::Equals([IO.Path]::GetFullPath((Get-Location).Path),$repo,[StringComparison]::OrdinalIgnoreCase)){
    throw 'Run from the repository root with an explicit working directory'
}
if(!$DeadlineUtc.EndsWith('Z')){throw 'New absolute UTC deadline ending in Z required'}
$end=[DateTimeOffset]::Parse($DeadlineUtc)
$budget=($end-[DateTimeOffset]::UtcNow).TotalSeconds
if($budget -lt 1860 -or $budget -gt 36000){throw 'New budget must be31min..10h'}
foreach($name in @('Root','Preparation','Qualification')){
    $value=Get-Variable -Name $name -ValueOnly
    if(![IO.Path]::IsPathFullyQualified($value) -or $value -notmatch '^[A-Za-z]:[\\/]' -or $value -match '[\x00-\x1f"''`$;&|<>*?(){}!]'){
        throw "$name must be an absolute Windows drive path"
    }
    Set-Variable -Name $name -Value ([IO.Path]::GetFullPath($value))
}
if((Test-Path -LiteralPath $Root) -or [IO.Path]::GetPathRoot($Root) -eq $Root){throw 'Fresh non-root output path required'}
foreach($name in @('LinuxPython','PrivateBase','V8Deployment','V8Manifest')){
    $value=Get-Variable -Name $name -ValueOnly
    if(!$value.StartsWith('/') -or $value -match '[\x00-\x1f"''\\`$;&|<>*?(){}!]' -or $value -match '(^|/)\.\.(/|$)'){
        throw "$name must be an absolute Linux path without traversal or control characters"
    }
}
if(!$WslDistro -or $WslDistro -match '[\x00-\x1f"''\\`$;&|<>*?(){}!]'){throw 'Explicit WSL distribution required'}
& wsl.exe -d $WslDistro -u root -- test -x $LinuxPython
if($LASTEXITCODE -ne 0){throw 'LinuxPython must be an existing executable'}
& wsl.exe -d $WslDistro -u root -- test -d $PrivateBase
if($LASTEXITCODE -ne 0){throw 'PrivateBase must be an existing Linux directory'}
& wsl.exe -d $WslDistro -u root -- test -d $V8Deployment
if($LASTEXITCODE -ne 0){throw 'V8Deployment must be an existing Linux directory'}
& wsl.exe -d $WslDistro -u root -- test -f $V8Manifest
if($LASTEXITCODE -ne 0){throw 'V8Manifest must be an existing Linux file'}
function Linux-Path([string]$path){
    # Forward slashes avoid WSL interpreting Windows separators as escapes.
    $result=(& wsl.exe -d $WslDistro -u root -- wslpath -a -- $path.Replace('\','/'))
    if($LASTEXITCODE -ne 0 -or !$result -or !$result.StartsWith('/')){throw 'wslpath failed'}
    return $result
}
$linuxRoot=Linux-Path $Root
$linuxPreparation=Linux-Path $Preparation
$linuxQualification=Linux-Path $Qualification
$proof=Get-Content -LiteralPath "$Qualification/result.json" -Raw|ConvertFrom-Json
$receipt=Get-Content -LiteralPath "$Preparation/preparation.json" -Raw|ConvertFrom-Json
if($proof.status -ne 'PASS_FULL_START_BC' -or $proof.cleanupErrors.Count -ne 0 -or $proof.trials.Count -ne 3 -or $proof.checkpoint_sha256 -ne $receipt.checkpoint_sha256){throw 'Qualified initializer required'}
$deploymentHash=[string]$proof.deployment_manifest_sha256
if($deploymentHash -cnotmatch '^[0-9a-f]{64}$'){throw 'Fresh V8 deployment-bound qualification required'}
$running=Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'wsl.exe'"
if($running|Where-Object {$_.CommandLine -match 'record_rl1168.(supervise|train|compare)|vision_qr1167.run|vision_qr1167.*entry.py|tmrl_residual1164.*run.py|campaign_entry.*--role.*learner'}){throw 'Inspect existing campaign before new launch'}
$guard=Get-Content -LiteralPath "$repo/tools/native_line1134/keep_awake.ps1" -Raw
$guard=$guard.Replace('Owned bounded headless native line search; release with its exact Windows owner','Owned bounded full-control QR; release with exact Windows owner')
if(!(Test-Path -LiteralPath (Join-Path $PSHOME 'pwsh.exe'))){throw 'PowerShell executable for guard is unavailable'}
New-Item -ItemType Directory -Path "$Root/source/tools/record_rl1168","$Root/source/tools/vision_qr1167" -Force|Out-Null
Copy-Item -LiteralPath (Get-ChildItem -LiteralPath "$repo/tools/record_rl1168" -Filter '*.py').FullName -Destination "$Root/source/tools/record_rl1168"
Copy-Item -LiteralPath "$repo/tools/vision_qr1167/returns.py" -Destination "$Root/source/tools/vision_qr1167"
# Preserve the checkout used to verify the deployment independently in every
# child interpreter. Bytecode is deliberately excluded from this snapshot.
$sdkRoot=Join-Path $repo 'trackmaniarl'
Get-ChildItem -LiteralPath $sdkRoot -Recurse -Force|ForEach-Object {
 if(($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'SDK snapshot cannot contain symlinks'}
 if(!$_.PSIsContainer -and $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -notin @('.pyc','.pyo')){
  $relative=$_.FullName.Substring($sdkRoot.Length+1)
  $destination=Join-Path "$Root/source/trackmaniarl" $relative
  [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($destination))|Out-Null
  Copy-Item -LiteralPath $_.FullName -Destination $destination
 }
}
[IO.File]::WriteAllText("$Root/source/tools/__init__.py",'')
[IO.File]::WriteAllText("$Root/source/tools/vision_qr1167/__init__.py",'')
$hashes=@{}
Get-ChildItem -LiteralPath "$Root/source" -Recurse -File|ForEach-Object {
 $relative=$_.FullName.Substring(("$Root/source").Length+1).Replace('\','/')
 $hashes[$relative]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
}
[IO.File]::WriteAllText("$Root/source-sha256.json",($hashes|ConvertTo-Json))
$launch=@{preparation=$linuxPreparation;qualification=$linuxQualification;deadline_utc=$DeadlineUtc;private_base=$PrivateBase;wsl_distro=$WslDistro;linux_python=$LinuxPython;deployment_manifest_sha256=$deploymentHash;v8_deployment=$V8Deployment;v8_manifest=$V8Manifest}
[IO.File]::WriteAllText("$Root/launch.json",($launch|ConvertTo-Json))
$module=if($Comparison){'tools.record_rl1168.compare'}else{'tools.record_rl1168.supervise'}
# WSL may invoke a Linux shell for the command line. Reject shell metacharacters
# before invoking it and quote arguments containing whitespace.
function Quote-Argument([string]$value){
    if($value -match '[\x00-\x1f"''\\`$;&|<>*?(){}!]'){throw 'Unsupported character in WSL argument'}
    if($value -match '\s'){return '"'+$value+'"'}
    return $value
}
$arguments=@('-d',$WslDistro,'-u','root','--cd',"$linuxRoot/source",'--','/usr/bin/env',"TMRL_CAMPAIGN_PRIVATE_BASE=$PrivateBase","TMRL_V8_DEPLOYMENT=$V8Deployment","TMRL_V8_MANIFEST=$V8Manifest","TMRL_V8_MANIFEST_SHA256=$deploymentHash",$LinuxPython,'-B','-m',$module,$linuxRoot)|ForEach-Object {Quote-Argument $_}
$owner=$null;$power=$null;$record=$null
function Same-Process([int]$processId,[string]$birth){
    $process=Get-Process -Id $processId -ErrorAction SilentlyContinue
    return $null -ne $process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $birth
}
try {
    $owner=Start-Process wsl.exe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/parent.stdout" -RedirectStandardError "$Root/parent.stderr"
    $record=@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startedUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;deadlineUtc=$DeadlineUtc}
    [IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
    [IO.File]::WriteAllText("$Root/keep_awake.ps1",$guard)
    $guardFile='"'+"$Root/keep_awake.ps1"+'"';$guardRoot='"'+$Root+'"'
    $power=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',$guardFile,'-Root',$guardRoot) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
    $powerBirth=$power.StartTime.ToUniversalTime().Ticks.ToString()
    [IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$power.Id;birthUtcTicks=$powerBirth}|ConvertTo-Json))
    $ready=$false;$waitUntil=[DateTime]::UtcNow.AddSeconds(15)
    while([DateTime]::UtcNow -lt $waitUntil){
        if(!(Same-Process $record.pid $record.birthUtcTicks) -or !(Same-Process $power.Id $powerBirth)){throw 'Owner or power guard exited before readiness'}
        if(Test-Path -LiteralPath "$Root/power-guard-state.json"){
            try {$state=Get-Content -LiteralPath "$Root/power-guard-state.json" -Raw|ConvertFrom-Json} catch {$state=$null}
            if($state -and $state.phase -eq 'active' -and $state.pid -eq $power.Id -and [string]$state.birthUtcTicks -eq $powerBirth -and $state.ownerPid -eq $record.pid -and [string]$state.ownerBirthUtcTicks -eq $record.birthUtcTicks -and $state.displayRequired -eq $false -and @($state.heldKinds).Count -eq 2 -and 1 -in $state.heldKinds -and 3 -in $state.heldKinds){$ready=$true;break}
        }
        Start-Sleep -Milliseconds 250
    }
    if(!$ready){throw 'Power guard did not report matching active requests'}
    $record|ConvertTo-Json
} catch {
    $failure=$_
    if($owner){
        try {[IO.File]::WriteAllText("$Root/STOP",'launcher startup failure')} catch {Write-Warning "Could not request cooperative stop: $_"}
        $until=[DateTime]::UtcNow.AddSeconds(40)
        while([DateTime]::UtcNow -lt $until){
            if($record -and !(Same-Process $record.pid $record.birthUtcTicks)){break}
            if(!$record -and $owner.HasExited){break}
            Start-Sleep -Milliseconds 250
        }
        if(($record -and (Same-Process $record.pid $record.birthUtcTicks)) -or (!$record -and !$owner.HasExited)){
            Write-Warning 'The exact owner has not yet exited; no process was forcibly terminated'
        }
    }
    throw $failure
}
