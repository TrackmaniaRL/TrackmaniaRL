param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root)
foreach($name in @('launch-intent.json','windows-launch.json','state.json','parent-launch-marker')){
 if(Test-Path -LiteralPath "$Root/$name"){throw 'Never duplicate an existing launch; inspect its full identities first'}
}
$linuxRoot='/mnt/f/'+$Root.Substring(3).Replace('\','/')
$python='/home/kuba/tmrl-full-qr-20261022-011/vision-optimization826/renderer-venv/bin/python'
[IO.File]::WriteAllText("$Root/launch-intent.json",(@{utc=[DateTime]::UtcNow.ToString('o');linuxRoot=$linuxRoot;newCampaign=$true}|ConvertTo-Json))
$owner=Start-Process wsl.exe -ArgumentList @('-d','Ubuntu','-u','root','--',$python,'-B',"$linuxRoot/source/search.py",$linuxRoot) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/windows-parent.stdout" -RedirectStandardError "$Root/windows-parent.stderr"
$record=[ordered]@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root;linuxRoot=$linuxRoot;gameStarted=$false}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$guard=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$guard.Id;birthUtcTicks=$guard.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
