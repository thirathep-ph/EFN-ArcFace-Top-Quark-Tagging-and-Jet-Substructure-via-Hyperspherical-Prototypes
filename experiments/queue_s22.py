import time, subprocess, pathlib, sys
log42 = pathlib.Path("experiments/robustness_s22_m05/train.log")
err42 = pathlib.Path("experiments/robustness_s22_m05/train.err")
print("[queue] waiting for seed42 (chunk mode, scheduler fix +500)...", flush=True)
last_mtime = 0
stall = 0
while True:
    if log42.exists():
        try:
            txt = log42.read_text(errors="ignore")
            err = err42.read_text(errors="ignore") if err42.exists() else ""
            if "[COMPLETE]" in txt:
                print("[queue] seed42 COMPLETE", flush=True)
                break
            if "ValueError" in txt or "ValueError" in err or "Traceback" in err:
                print("[queue] seed42 FAILED - ValueError/Traceback detected, breaking to avoid hang", flush=True)
                print(err[-2000:], flush=True)
                break
            mtime = log42.stat().st_mtime
            if mtime == last_mtime:
                stall += 1
            else:
                stall = 0
                last_mtime = mtime
            if stall >= 15:
                print("[queue] seed42 STALL 15min no log update - breaking", flush=True)
                break
        except Exception as e:
            print(f"[queue] read error {e}", flush=True)
    time.sleep(60)
print("[queue] starting seed 123 (chunk mode)", flush=True)
subprocess.run([sys.executable, "-u", "scripts/train_arcface.py", "--scale", "22", "--seed", "123", "--checkpoint_dir", "experiments/robustness_s22_seed_123", "--output_dir", "experiments/robustness_s22_seed_123"])
print("[queue] seed123 done", flush=True)
print("[queue] starting seed 7 (chunk mode)", flush=True)
subprocess.run([sys.executable, "-u", "scripts/train_arcface.py", "--scale", "22", "--seed", "7", "--checkpoint_dir", "experiments/robustness_s22_seed_7", "--output_dir", "experiments/robustness_s22_seed_7"])
print("[queue] all done", flush=True)
