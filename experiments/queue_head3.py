import time, subprocess, pathlib, sys
qlog = pathlib.Path("experiments/s22_queue.log")
errlog = pathlib.Path("experiments/robustness_s22_m05/train.err")
print("[head3-queue] waiting for s=22 3-seed queue to finish...", flush=True)
stall = 0
last = 0
while True:
    if qlog.exists():
        try:
            txt = qlog.read_text(errors="ignore")
            if "all done" in txt.lower():
                print("[head3-queue] s=22 queue done detected", flush=True)
                break
            if "FAILED" in txt or "STALL" in txt:
                print("[head3-queue] upstream FAILED/STALL - proceeding anyway to salvage head variance", flush=True)
                break
            mtime = qlog.stat().st_mtime
            if mtime == last:
                stall += 1
            else:
                stall = 0
                last = mtime
            if stall >= 30:
                print("[head3-queue] qlog STALL 30min - proceeding", flush=True)
                break
        except Exception as e:
            print(f"[head3-queue] read error {e}", flush=True)
    time.sleep(60)

base = "experiments/robustness_s16_m05"
jobs = [
    (f"linear_seed123", f"experiments/baselines_linear_seed_123"),
    (f"linear_seed7", f"experiments/baselines_linear_seed_7"),
    (f"coslinear_seed123", f"experiments/baselines_coslinear_seed_123"),
    (f"coslinear_seed7", f"experiments/baselines_coslinear_seed_7"),
]
for tag, comp in jobs:
    print(f"[head3-queue] running head geometry {tag} vs {comp}", flush=True)
    cmd = [sys.executable, "scripts/analysis/compute_head_geometry.py", "--model_dir", base, "--compare_dir", comp, "--out_tag", f"_{tag}", "--max_events", "50000"]
    print(f"[head3-queue] cmd: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[head3-queue] failed {tag}: {e}", flush=True)
    print(f"[head3-queue] done {tag}", flush=True)

print("[head3-queue] all head-geometry variance done", flush=True)
