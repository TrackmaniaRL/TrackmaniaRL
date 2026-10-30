param([Parameter(Mandatory=$true)][string]$Root,
      [Parameter(Mandatory=$true)][string]$Preparation,
      [int]$MaxSeconds=21600,
      [int]$MaxTransitions=100000,
      [string]$ResumeCheckpoint='',
      [string]$StepTimingProof='',
      [string]$Initialization='',
      [string]$RetentionDataset='',
      [string]$CleanupRecovery='',
      [string]$DeadlineUtc='')
$ErrorActionPreference='Stop'
$repo='C:/Users/szulc/.codex/worktrees/headless-training-parity/AITrackmania'
if((Get-Location).Path.Replace('\','/') -ne $repo){throw 'Explicit approved workdir required'}
if(!$DeadlineUtc -or !$DeadlineUtc.EndsWith('Z')){throw 'Explicit absolute UTC deadline ending in Z required'}
$absoluteDeadline=[DateTimeOffset]::Parse($DeadlineUtc,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::AdjustToUniversal)
# Compare UTC offsets explicitly: DateTime parsing may convert Z to local time.
# Leave two minutes for the final checkpoint and bounded environment cleanup.
$deadlineBudget=[int][Math]::Floor(($absoluteDeadline-[DateTimeOffset]::UtcNow).TotalSeconds)-120
$MaxSeconds=[Math]::Min($MaxSeconds,$deadlineBudget)
if($MaxSeconds -lt 60 -or $MaxSeconds -gt 21600){throw 'Budget must be 60..21600 seconds'}
if(Test-Path -LiteralPath $Root){throw 'Fresh run directory required'}
$reference=Join-Path $Preparation 'reward-reference-04/result.json'
$proof=Get-Content -LiteralPath $reference -Raw|ConvertFrom-Json
if($proof.status -ne 'PASS_REWARD_REFERENCE' -or $proof.error -or $proof.cleanup_error){throw 'Reference/reset admission failed'}
$running=Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'python3.12' OR Name = 'wsl.exe'"
if($running|Where-Object { $_.CommandLine -match 'vision_qr1167.run|vision_qr1167.*entry.py|campaign_entry.*--role.*learner|tmrl_residual1164.*run.py' }){throw 'Existing campaign process requires inspection'}
New-Item -ItemType Directory -Path "$Root/source/tools/vision_qr1167" -Force|Out-Null
[IO.File]::WriteAllText("$Root/source/tools/__init__.py",'')
[IO.File]::WriteAllText("$Root/source/tools/vision_qr1167/__init__.py",'')
Copy-Item -LiteralPath (Get-ChildItem -LiteralPath "$repo/tools/vision_qr1167" -Filter '*.py').FullName -Destination "$Root/source/tools/vision_qr1167"
$plan=[ordered]@{
 installation='C:/Users/szulc/AppData/Local/TICK/versions/2.2.0'
 dependencies='C:/Users/szulc/.codex/visualizations/2026/10/28/01a181c3-af49-7011-9fab-d6dceab9cf5b/tick-rea/api/python-deps'
 calibration=(Join-Path $Preparation 'camera-clock-01')
 reference=$reference
 reference_sha256=(Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash.ToLower()
 updates_per_transition=4
 device='cuda'; max_seconds=$MaxSeconds; max_transitions=$MaxTransitions
 absolute_deadline_utc=$absoluteDeadline.UtcDateTime.ToString('o')
 deadline_cleanup_reserve_seconds=120
 map_uid='oqIJ5rQDRrNwLPTh9H2p_W4tLof'
 algorithm='SDK Double QR-DQN, 64 quantiles, real camera 4x84x84'
 return_semantics='Finite driving attempts: goal or reset condition ends bootstrap; operator stop is not synthesized as a terminal transition'
 n_step_max=5
 replay_episode_provenance='Legacy untagged rows remain one-step; tagged real episodes never cross reset boundaries'
 seed=1167; previous_checkpoint_architecture_compatible=$false
 curriculum='50% real reset without teacher; 50% real teacher prefix to 2..16s; measured handoff and steering perturbations'
 evaluation_required='Separate frozen policy full-start evaluation; training suffixes never count as records'
}
if($StepTimingProof){
 $timing=Get-Content -LiteralPath $StepTimingProof -Raw|ConvertFrom-Json
 if($timing.status -ne 'PASS_EXACT_STEPS_CAMERA_AND_INPUTS' -or $timing.cleanup_error -or $timing.physics_steps -lt 330){throw 'Step timing proof not admitted'}
 if($timing.bridge_sha256 -ne (Get-FileHash -LiteralPath "$repo/tools/vision_qr1167/bridge.py" -Algorithm SHA256).Hash.ToLower()){throw 'Timing proof belongs to different bridge source'}
 $plan.fast_steps=$true
 $plan.step_timing_proof=$StepTimingProof
 $plan.step_timing_proof_sha256=(Get-FileHash -LiteralPath $StepTimingProof -Algorithm SHA256).Hash.ToLower()
}
if($RetentionDataset){
 $receipt=Get-Content -LiteralPath "$RetentionDataset/receipt.json" -Raw|ConvertFrom-Json
 if($receipt.status -ne 'ADMITTED_TEACHER_WINDOWS' -or $receipt.dataset_sha256 -ne (Get-FileHash -LiteralPath "$RetentionDataset/demonstrations.npz" -Algorithm SHA256).Hash.ToLower()){throw 'Retention dataset not admitted'}
 $plan.retention_dataset=$RetentionDataset
 if($receipt.parent_dataset_sha256){$plan.admit_extended_teacher_actions=$true; $plan.data_extension='Exact parent rows and separately admitted additions verified before any update; preserve weights and optimizer'}
 $plan.auxiliary_imitation='decaying supervised margin; counted separately from QR reward updates'
}
if($Initialization){
 if($ResumeCheckpoint){throw 'Choose initialization or resume'}
 $bc=Get-Content -LiteralPath ((Split-Path $Initialization -Parent)+'/result.json') -Raw|ConvertFrom-Json
 $digest=(Get-FileHash -LiteralPath $Initialization -Algorithm SHA256).Hash.ToLower()
 if($null -eq $bc.max_abs_q -or $bc.max_abs_q -gt 20){throw 'BC Q-scale admission missing or failed'}
 if($bc.status -ne 'BC_INITIALIZATION_ONLY' -or $bc.rl_updates -ne 0 -or $bc.policy_sha256 -ne $digest){throw 'BC initialization provenance mismatch'}
 $plan.resume_checkpoint=$Initialization
 $plan.resume_sha256=$digest
 $plan.initialization_kind='supervised camera BC; zero RL updates'
 $plan.initialization_bc_updates=$bc.bc_updates
 if($bc.policy_action_ids){$plan.policy_action_ids=$bc.policy_action_ids; $plan.control_space=$bc.control_space}
 $plan.epsilon_initial=0.10
 $plan.epsilon_minimum=0.03
 $plan.epsilon_decay_transitions=10000
}
if($ResumeCheckpoint){
 $previous=Split-Path $ResumeCheckpoint -Parent
 $previousPlan=Get-Content -LiteralPath "$previous/plan.json" -Raw|ConvertFrom-Json
 foreach($key in @('epsilon_initial','epsilon_minimum','epsilon_decay_transitions')){if($null -ne $previousPlan.$key){$plan[$key]=$previousPlan.$key}}
 $closed=Get-Content -LiteralPath "$previous/completion.json" -Raw|ConvertFrom-Json
 if($closed.cleanupErrors.Count -ne 0){
  if(!$CleanupRecovery){throw 'Previous cleanup incomplete'}
  $repair=Get-Content -LiteralPath $CleanupRecovery -Raw|ConvertFrom-Json
  foreach($key in @('status','completion_sha256','campaign','owner_pid','owner_birthUtcTicks','guard_pid','guard_birthUtcTicks','cleanupErrors','remainingOwned','map_uid','effective_speed','teacher_execution','verified_utc')){
   if($key -notin $repair.psobject.Properties.Name){throw "Cleanup recovery missing $key"}
  }
  $repairAge=([DateTime]::UtcNow-([DateTime]$repair.verified_utc).ToUniversalTime()).TotalSeconds
  if($repairAge -gt 120 -or $repairAge -lt -5){throw 'Cleanup recovery must be freshly verified'}
  $completionDigest=(Get-FileHash -LiteralPath "$previous/completion.json" -Algorithm SHA256).Hash.ToLower()
  if($repair.status -ne 'PASS_CLEANUP_RECOVERY' -or $repair.completion_sha256 -ne $completionDigest -or
     $repair.campaign.Replace('\','/') -ne $previous.Replace('\','/') -or
     $repair.cleanupErrors.Count -ne 0 -or $repair.remainingOwned.Count -ne 0 -or
     $repair.map_uid -ne $plan.map_uid -or $repair.effective_speed -ne 0 -or $repair.teacher_execution -ne $false){throw 'Cleanup recovery not admitted'}
  $plan.cleanup_recovery=$CleanupRecovery
  $plan.cleanup_recovery_sha256=(Get-FileHash -LiteralPath $CleanupRecovery -Algorithm SHA256).Hash.ToLower()
 }
 $priorOwner=Get-Content -LiteralPath "$previous/windows-launch.json" -Raw|ConvertFrom-Json
 if($closed.cleanupErrors.Count -ne 0 -and ($repair.owner_pid -ne $priorOwner.pid -or $repair.owner_birthUtcTicks -ne $priorOwner.birthUtcTicks)){throw 'Cleanup recovery owner mismatch'}
 if($closed.cleanupErrors.Count -ne 0){
  $priorGuard=Get-Content -LiteralPath "$previous/power-guard-launch.json" -Raw|ConvertFrom-Json
  if($repair.guard_pid -ne $priorGuard.pid -or $repair.guard_birthUtcTicks -ne $priorGuard.birthUtcTicks){throw 'Cleanup recovery guard mismatch'}
  $guardProcess=Get-Process -Id $priorGuard.pid -ErrorAction SilentlyContinue
  if($guardProcess -and $guardProcess.StartTime.ToUniversalTime().Ticks.ToString() -eq $priorGuard.birthUtcTicks){throw 'Previous guard still alive'}
 }
 $process=Get-Process -Id $priorOwner.pid -ErrorAction SilentlyContinue
 if($process -and $process.StartTime.ToUniversalTime().Ticks.ToString() -eq $priorOwner.birthUtcTicks){throw 'Previous owner still alive'}
 $plan.resume_checkpoint=$ResumeCheckpoint
 $plan.resume_sha256=(Get-FileHash -LiteralPath $ResumeCheckpoint -Algorithm SHA256).Hash.ToLower()
}
[IO.File]::WriteAllText("$Root/plan.json",($plan|ConvertTo-Json -Depth 5))
$entry=@"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append('$repo')
from tools.vision_qr1167.run import main
main()
"@
[IO.File]::WriteAllText("$Root/source/entry.py",$entry)
$hashes=Get-ChildItem -LiteralPath "$Root/source/tools/vision_qr1167" -Filter '*.py'|ForEach-Object {@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()}}
[IO.File]::WriteAllText("$Root/source-sha256.json",($hashes|ConvertTo-Json))
$python='H:/Studia/inzynierskie/inzynierkav2/AITrackmania/.venv/Scripts/python.exe'
$owner=Start-Process -FilePath $python -ArgumentList @('-B',"$Root/source/entry.py",$Root) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/parent.stdout" -RedirectStandardError "$Root/parent.stderr"
$record=@{pid=$owner.Id;birthUtcTicks=$owner.StartTime.ToUniversalTime().Ticks.ToString();startedUtc=$owner.StartTime.ToUniversalTime().ToString('o');root=$Root}
[IO.File]::WriteAllText("$Root/windows-launch.json",($record|ConvertTo-Json))
$guard=Get-Content -LiteralPath "$repo/tools/native_line1134/keep_awake.ps1" -Raw
$guard=$guard.Replace('Owned bounded headless native line search; release with its exact Windows owner','Owned bounded camera QR-DQN; release with its exact Windows owner').Replace('AddHours(10).AddMinutes(10)','AddHours(6).AddMinutes(10)').Replace('@(1,3)','@(0,1,3)').Replace('displayRequired=$false','displayRequired=$true')
[IO.File]::WriteAllText("$Root/keep_awake.ps1",$guard)
$power=Start-Process (Join-Path $PSHOME 'pwsh.exe') -ArgumentList @('-NoProfile','-File',"$Root/keep_awake.ps1",'-Root',$Root) -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput "$Root/power-guard.stdout" -RedirectStandardError "$Root/power-guard.stderr"
[IO.File]::WriteAllText("$Root/power-guard-launch.json",(@{pid=$power.Id;birthUtcTicks=$power.StartTime.ToUniversalTime().Ticks.ToString()}|ConvertTo-Json))
$record|ConvertTo-Json
