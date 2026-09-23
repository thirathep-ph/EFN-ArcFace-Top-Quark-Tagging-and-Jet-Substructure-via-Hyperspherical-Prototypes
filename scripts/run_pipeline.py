"""
Reproducible Pipeline Orchestrator
===================================
Runs the full Jet Tagging pipeline from training through paper figure generation.

Usage
-----
    python scripts/run_pipeline.py          # full pipeline
    python scripts/run_pipeline.py --train      # train only
    python scripts/run_pipeline.py --baselines  # baselines only
    python scripts/run_pipeline.py --figures    # paper figures only

Each step can also be run individually via its dedicated script.
"""
import os, sys, subprocess, argparse

from arcefn.utils.paths import SCRIPTS_DIR


def _run(script, label, *args):
    path = SCRIPTS_DIR / script
    cmd = [sys.executable, str(path)] + list(args)
    print(f"\n{'='*60}")
    print(f"  [{label}] Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR.parent)
    if result.returncode != 0:
        print(f"  [FAILED] {script} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  [OK] {script} finished.\n")


def train():
    _run("train_arcface.py", "TRAIN")
    _run("train_baselines.py", "BASELINES")


def evaluate():
    _run("evaluate_model.py", "EVAL", "--split", "test")
    _run("evaluate_robustness.py", "ROBUSTNESS EVAL")


def paper_figures():
    _run("analysis/compute_eigengap.py", "FIG 2")
    _run("analysis/compute_mass_vs_score.py", "FIG 10")
    _run("analysis/compute_dglap_fit.py", "FIG 8 / TABLE 2")
    _run("analysis/compute_bootstrap_ci.py", "FIG 4/12 / TABLE 1", "--max-events", "404000")
    _run("analysis/compute_interpretation.py", "FIG 13 / TABLE 3")
    _run("analysis/compute_saliency.py", "FIG 14")
    _run("analysis/compute_dimensionality.py", "TABLE 3")
    _run("analysis/compute_robustness_tests.py", "TABLE 4")
    _run("analysis/generate_paper_figures.py", "FIG 3/5/6/9/11")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the full Jet Tagging pipeline")
    p.add_argument("--train", action="store_true", help="Train models only")
    p.add_argument("--evaluate", action="store_true", help="Evaluate models only")
    p.add_argument("--figures", action="store_true", help="Regenerate paper figures only")
    args = p.parse_args()

    if not any([args.train, args.evaluate, args.figures]):
        args.train = args.evaluate = args.figures = True

    if args.train:
        train()
    if args.evaluate:
        evaluate()
    if args.figures:
        paper_figures()

    print("\nDone. All requested pipeline steps completed.")
