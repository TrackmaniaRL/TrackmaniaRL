param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference='Stop'
$launch=Get-Content -LiteralPath "$Root/windows-launch.json" -Raw|ConvertFrom-Json
Add-Type -TypeDefinition @'
using System;using System.Runtime.InteropServices;
public static class NativeSearchPower {
 [StructLayout(LayoutKind.Sequential,CharSet=CharSet.Unicode)] public struct Reason {public uint Version;public uint Flags;[MarshalAs(UnmanagedType.LPWStr)]public string Text;}
 [DllImport("kernel32.dll",SetLastError=true)] public static extern IntPtr PowerCreateRequest(ref Reason reason);
 [DllImport("kernel32.dll",SetLastError=true)] [return:MarshalAs(UnmanagedType.Bool)]public static extern bool PowerSetRequest(IntPtr handle,int kind);
 [DllImport("kernel32.dll")] [return:MarshalAs(UnmanagedType.Bool)]public static extern bool PowerClearRequest(IntPtr handle,int kind);
 [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
}
'@
$reason=New-Object NativeSearchPower+Reason
$reason.Flags=1;$reason.Text='Owned bounded headless native line search; release with its exact Windows owner'
$handle=[NativeSearchPower]::PowerCreateRequest([ref]$reason)
if($handle -eq [IntPtr]::Zero -or $handle -eq [IntPtr]::new(-1)){throw 'Power request creation failed'}
$held=@()
$self=Get-Process -Id $PID
$state=[ordered]@{phase='starting';pid=$PID;birthUtcTicks=$self.StartTime.ToUniversalTime().Ticks.ToString();ownerPid=$launch.pid;ownerBirthUtcTicks=$launch.birthUtcTicks;displayRequired=$false}
function Save-State {
 [IO.File]::WriteAllText("$Root/power-guard-state.stage",($state|ConvertTo-Json -Depth 5))
 Move-Item -LiteralPath "$Root/power-guard-state.stage" -Destination "$Root/power-guard-state.json" -Force
}
try {
 foreach($kind in @(1,3)){
  if(-not [NativeSearchPower]::PowerSetRequest($handle,$kind)){throw "Power request failed $kind"}
  $held+= $kind
 }
 $state.phase='active';$state.heldKinds=$held;Save-State
 $safetyDeadline=[DateTime]::UtcNow.AddHours(10).AddMinutes(10)
 while([DateTime]::UtcNow -lt $safetyDeadline){
  $owner=Get-Process -Id $launch.pid -ErrorAction SilentlyContinue
  if($null -eq $owner -or $owner.StartTime.ToUniversalTime().Ticks.ToString() -ne [string]$launch.birthUtcTicks){break}
  if(Test-Path -LiteralPath "$Root/RELEASE_AWAKE"){break}
  Start-Sleep -Seconds 5
 }
}catch{$state.error=$_.Exception.Message;throw}
finally{
 foreach($kind in $held){[void][NativeSearchPower]::PowerClearRequest($handle,$kind)}
 [void][NativeSearchPower]::CloseHandle($handle)
 $state.phase='released';$state.releasedUtc=[DateTime]::UtcNow.ToString('o');Save-State
}
