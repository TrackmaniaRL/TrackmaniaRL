param(
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Output,
    [string]$Distribution = 'Ubuntu',
    [string]$User = 'root',
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$arguments = @('-d', $Distribution, '-u', $User, '--cd', $Repository, '--', 'env', 'WANDB_MODE=offline', $Python,
               '-m', 'tools.headless_campaign_queue.campaign', $Manifest, '--output', $Output)

# Always preflight before requesting power or starting a process.
& wsl.exe @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $Execute) { exit 0 }

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class CampaignPowerRequest {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint state);
}
'@
$continuous = [uint32]2147483648
$required = [uint32]2147483649
if ([CampaignPowerRequest]::SetThreadExecutionState($required) -eq 0) {
    throw 'Unable to prevent Windows idle sleep. No campaign was started.'
}
try {
    & wsl.exe @arguments '--execute'
    $result = $LASTEXITCODE
} finally {
    [void][CampaignPowerRequest]::SetThreadExecutionState($continuous)
}
exit $result
