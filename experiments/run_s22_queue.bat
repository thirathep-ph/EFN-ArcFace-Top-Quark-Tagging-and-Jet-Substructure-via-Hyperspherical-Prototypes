@echo off
cd /d D:\research\jet_tagging
D:\research\jet_tagging\.venv\Scripts\python.exe -u experiments/queue_s22.py > experiments/s22_queue.log 2>&1
