$ErrorActionPreference = "Continue"
function Wait-ForSeed42 {
    Write-Output "[queue] waiting for seed42 (robustness_s22_m05) to finish..."
    while ($true) {
        $log = "D:\research\jet_tagging\experiments\robustness_s22_m05\train.log"
        if (Test-Path $log) {
            $hit = Select-String -Path $log -Pattern "\[COMPLETE\]" -Quiet -ErrorAction SilentlyContinue
            if ($hit) { Write-Output "[queue] seed42 COMPLETE detected"; break }
        }
        # also check if process still alive - if log stale and no python, break
        Start-Sleep -Seconds 60
    }
}

# If seed42 still running, wait; otherwise proceed immediately if already done
Wait-ForSeed42

Write-Output "[queue] starting seed 123 s=22"
& "D:\research\jet_tagging\.venv\Scripts\python.exe" "D:\research\jet_tagging\scripts\train_arcface.py" --scale 22 --seed 123 --checkpoint_dir "experiments/robustness_s22_seed_123" --output_dir "experiments/robustness_s22_seed_123"
Write-Output "[queue] seed 123 exit code $LASTEXITCODE"

Write-Output "[queue] starting seed 7 s=22"
& "D:\research\jet_tagging\.venv\Scripts\python.exe" "D:\research\jet_tagging\scripts\train_arcface.py" --scale 22 --seed 7 --checkpoint_dir "experiments/robustness_s22_seed_7" --output_dir "experiments/robustness_s22_seed_7"
Write-Output "[queue] seed 7 exit code $LASTEXITCODE"
Write-Output "[queue] all s=22 seeds done"
