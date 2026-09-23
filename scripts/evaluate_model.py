"""
Single-Model Evaluation Script for EFN + ArcFace
=================================================
Evaluates a trained model checkpoint on a specified dataset split (train,
val, or test) and reports accuracy, precision, recall, F1, and confusion
matrix under both ``eval`` and ``train`` Batch-Normalisation modes.

Usage
-----
::

    python scripts/evaluate_model.py --model_path checkpoints/checkpoint.pt
    python scripts/evaluate_model.py --model_path checkpoints/checkpoint.pt --split test --max_events 10000
"""
import argparse
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from arcefn.utils.paths import DATA_DIR
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset


def evaluate_model(model_path, data_dir, split, max_events, mode):
    """
    Load a checkpoint and evaluate on the requested split.

    Parameters
    ----------
    model_path : str
        Path to ``checkpoint.pt``.
    data_dir : str or Path
        Directory containing ``train.h5`` / ``val.h5`` / ``test.h5``.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    max_events : int
        Maximum events to load.
    mode : str
        BN evaluation mode: ``'eval'``, ``'train'``, or ``'both'``.
    """
    device, _ = get_device()

    # 1. Load Model
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded checkpoint: {model_path}")

    # 2. Load Dataset
    file_path = DATA_DIR / f"{split}.h5"
    events, labels, weights = load_awkward(file_path, max_events=max_events)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # 3. Evaluation helper
    def run_eval(bn_mode):
        if bn_mode == 'train':
            model.train()
        else:
            model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y, w, m, _ in loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits, _, _ = model(x, mask=m)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = np.mean(all_preds == all_labels)
        tp = int(np.sum((all_preds == 1) & (all_labels == 1)))
        fp = int(np.sum((all_preds == 1) & (all_labels == 0)))
        fn = int(np.sum((all_preds == 0) & (all_labels == 1)))
        tn = int(np.sum((all_preds == 0) & (all_labels == 0)))
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0)
        unique, counts = np.unique(all_preds, return_counts=True)
        pred_dist = dict(zip(unique.tolist(), counts.tolist()))

        return {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'pred_dist': pred_dist,
        }

    # 4. Run
    modes_to_test = ['eval', 'train'] if mode == 'both' else [mode]

    print("\n" + "=" * 55)
    print(f"EVALUATION REPORT: {split.upper()} split ({len(dataset):,} events)")
    print("=" * 55)

    for m in modes_to_test:
        res = run_eval(m)
        print(f"\n[BN Mode: {m.upper()}]")
        print(f"  Accuracy:  {res['accuracy']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall:    {res['recall']:.4f}")
        print(f"  F1-Score:  {res['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True QCD (0):  TN={res['tn']:5d} | FP={res['fp']:5d}")
        print(f"    True Top (1):  FN={res['fn']:5d} | TP={res['tp']:5d}")
        print(f"  Prediction Distribution: {res['pred_dist']}")
    print("=" * 55 + "\n")


def main():
    """
    Parse arguments and evaluate the model.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate EFN + ArcFace model checkpoint")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved .pt checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/top_tagging',
                        help='Directory with HDF5 data splits')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--max_events', type=int, default=5000)
    parser.add_argument('--mode', type=str, default='both',
                        choices=['eval', 'train', 'both'],
                        help="BN mode: 'eval', 'train', or 'both'")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.split,
                   args.max_events, args.mode)


if __name__ == '__main__':
    main()
