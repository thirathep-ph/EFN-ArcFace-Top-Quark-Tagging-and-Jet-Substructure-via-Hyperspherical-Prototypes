#!/usr/bin/env python
"""
Run all d=2 analysis scripts by temporarily linking the d=2 model dir
to the expected canonical path, then running the original scripts.
"""
import os
import subprocess
import sys
import argparse

def run_script(script_name, model_dir, extra_args=None):
    """Run a script with the model_dir set."""
    cmd = [sys.executable, "-m", f"scripts.analysis.{script_name}"]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["ARCEFN_MODEL_DIR"] = model_dir
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all d=2 analysis scripts")
    parser.add_argument("--model_dir", type=str, default="experiments/ablation_dim_2",
                        help="Directory containing d=2 model checkpoint and embeddings")
    parser.add_argument("--scripts", nargs="+", default=None,
                        help="Specific scripts to run (default: all)")
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Model directory: {model_dir}")

    # Check that model dir exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist")
        return 1

    # Check for required files
    required = ["checkpoint.pt", "embeddings.npz", "history.json", "config.json"]
    for f in required:
        path = os.path.join(model_dir, f)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")

    # Scripts to run (in order)
    scripts = [
        "generate_paper_figures",      # Core paper figures (UMAP, Lund, physics, heatmap)
        "compute_subclass_lund_planes", # Subclass Lund planes (6 panels)
        "generate_hypersphere_viz",    # Hypersphere 3D visualization
        "generate_angular_collimation", # Angular collimation
        "compute_interpretation",       # MI, effect sizes, PCA
        "compute_bootstrap_ci",        # ROC, rejection, NMI/ARI
        "compute_head_geometry",       # Head geometry (DGLAP, counts)
        "compute_dglap_fit",           # DGLAP fits (Table 2)
        "compute_saliency",            # Saliency
        "compute_robustness_tests",    # Robustness tests
    ]

    if args.scripts:
        scripts = args.scripts

    failed = []
    for script in scripts:
        print(f"\n{'='*60}")
        print(f"Running {script} on d=2 model...")
        print(f"{'='*60}")
        success = run_script(script, args.model_dir)
        if not success:
            failed.append(script)
            print(f"FAILED: {script}")
        else:
            print(f"SUCCESS: {script}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(scripts)}")
    print(f"Passed: {len(scripts) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed scripts: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())