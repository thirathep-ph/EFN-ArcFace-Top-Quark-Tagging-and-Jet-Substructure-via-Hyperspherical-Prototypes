@echo off
cd /d D:\research\jet_tagging
D:\research\jet_tagging\.venv\Scripts\python.exe -u scripts/train_arcface.py --scale 22 --seed 42 --checkpoint_dir experiments/robustness_s22_m05 --output_dir experiments/robustness_s22_m05 > experiments/robustness_s22_m05/train.log 2>&1
