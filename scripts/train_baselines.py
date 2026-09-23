"""
Training Script: EFN Baselines (Linear + CosLinear)
====================================================
Trains and evaluates two ablation baselines:
    A: EFN + Linear + CE  (standard classifier)
    B: EFN + CosLinear + CE  (ArcFace without margin)

Uses chunked loading of the full datasets.

Usage:
    # Train Linear baseline (50 epochs)
    python scripts/train_baselines.py --model linear --epochs 50

    # Train CosLinear baseline (50 epochs)
    python scripts/train_baselines.py --model coslinear --epochs 30

    # Train both with 3 seeds each
    python scripts/train_baselines.py --model both --epochs 30 --seeds 42 123 7
"""
import json
import time
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcefn.data.loader import load_awkward, JetTaggingDataset, get_h5_len
from arcefn.models.efn_linear import TopTaggingLinear, count_parameters
from arcefn.models.efn_coslinear import TopTaggingCosLinear
from arcefn.utils.device import get_device, empty_cache, amp_available, get_amp_scaler
import gc


def train_epoch(model, loader, optimizer, device, scheduler=None, accumulation_steps=8, scaler=None):
    model.train()
    total_loss, total_acc = 0, 0
    n_batches = len(loader)
    optimizer.zero_grad()

    for batch_idx, (x, y, w, m, _) in enumerate(loader):
        x, y, m = x.to(device), y.to(device), m.to(device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, _ = model(x, mask=m)
                loss = F.cross_entropy(logits, y)
        else:
            logits, _ = model(x, mask=m)
            loss = F.cross_entropy(logits, y)

        loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
        else:
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        total_loss += loss.item() * accumulation_steps
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, file_path, device, backend_type='cpu', batch_size=1024, chunk_size=100000, cached_data=None):
    """Evaluate on full dataset using chunked loading (or cached data in no-chunk mode)."""
    model.eval()

    if cached_data is not None:
        # Use pre-loaded data (no-chunk mode) — avoids re-reading H5 every epoch
        events, labels, weights = cached_data
        ds = JetTaggingDataset(events, labels, weights)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        total_len = len(ds)
    else:
        total_len = get_h5_len(file_path)
        num_chunks = int(np.ceil(total_len / chunk_size))

    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        if cached_data is not None:
            # Single loader for cached data
            for x, y, w, m, _ in loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits, _ = model(x, mask=m)
                loss = F.cross_entropy(logits, y)
                total_loss += loss.item()
                n_batches += 1
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_logits.append(logits.cpu())
        else:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                n_events = min(chunk_size, total_len - start)

                events, labels, weights = load_awkward(
                    file_path, max_events=n_events, start_event=start, lazy=True
                )
                ds = JetTaggingDataset(events, labels, weights)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

                for x, y, w, m, _ in loader:
                    x, y, m = x.to(device), y.to(device), m.to(device)
                    logits, _ = model(x, mask=m)
                    loss = F.cross_entropy(logits, y)
                    total_loss += loss.item()
                    n_batches += 1
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    all_logits.append(logits.cpu())

                del loader, ds, events, labels, weights
                gc.collect()
                empty_cache(backend_type)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits).numpy()

    acc = np.mean(all_preds == all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_logits[:, 1])
    except Exception:
        auc = 0.0

    return {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1_score': float(f1),
        'roc_auc': float(auc), 'loss': total_loss / n_batches,
        'n_events': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear',
                        choices=['linear', 'coslinear', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds (e.g., 42 123 7)')
    parser.add_argument('--data_dir', type=str, default='data/top_tagging')
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/baselines')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--no-chunk', action='store_true', help='Disable chunked loading. Load entire dataset into RAM once (faster)')
    parser.add_argument('--chunk_size', type=int, default=100000,
                        help='Chunk size for reading datasets (only used when --no-chunk is not set)')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--particle_dim', type=int, default=128)
    parser.add_argument('--scale', type=float, default=16.0,
                        help='Scale factor (only for coslinear)')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision (FP16) for faster training on compatible GPUs')
    args = parser.parse_args()

    device, backend_type = get_device()
    use_amp = args.amp and amp_available(backend_type)
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")
    scaler = get_amp_scaler(backend_type, enabled=use_amp)

    train_path = os.path.join(args.data_dir, 'train.h5')
    val_path = os.path.join(args.data_dir, 'val.h5')
    test_path = os.path.join(args.data_dir, 'test.h5')

    train_len = get_h5_len(train_path)
    use_chunking = not args.no_chunk
    num_train_chunks = int(np.ceil(train_len / args.chunk_size)) if use_chunking else 1
    
    # Calculate scheduler steps (matching train_top.py's OneCycleLR setup)
    if use_chunking:
        total_train_batches = 0
        for chunk_idx in range(num_train_chunks):
            start = chunk_idx * args.chunk_size
            n_events = min(args.chunk_size, train_len - start)
            total_train_batches += int(np.ceil(n_events / args.batch_size))
    else:
        total_train_batches = int(np.ceil(train_len / args.batch_size))
    steps_per_epoch = int(np.ceil(total_train_batches / args.accumulation_steps))
    total_scheduler_steps = steps_per_epoch * args.epochs

    # Pre-load all data once (outside seed loop) to avoid re-reading H5 for each seed
    cached_train_data = None
    cached_val_data = None
    cached_test_data = None
    if not use_chunking:
        print("Pre-loading all data into memory (single-load mode)...")
        tr_ev, tr_lab, tr_w = load_awkward(
            train_path, max_events=train_len, lazy=True
        )
        cached_train_data = (tr_ev, tr_lab, tr_w)
        print(f"  Loaded {train_len:,} train events")
        val_len = get_h5_len(val_path)
        vl_ev, vl_lab, vl_w = load_awkward(
            val_path, max_events=val_len, lazy=True
        )
        cached_val_data = (vl_ev, vl_lab, vl_w)
        print(f"  Loaded {val_len:,} val events")
        test_len = get_h5_len(test_path)
        te_ev, te_lab, te_w = load_awkward(
            test_path, max_events=test_len, lazy=True
        )
        cached_test_data = (te_ev, te_lab, te_w)
        print(f"  Loaded {test_len:,} test events")

    models_to_train = ['linear', 'coslinear'] if args.model == 'both' else [args.model]

    for model_name in models_to_train:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Training: {model_name} | Seed: {seed}")
            print(f"{'='*60}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create model
            if model_name == 'linear':
                model = TopTaggingLinear(
                    embedding_dim=args.embedding_dim,
                    particle_dim=args.particle_dim
                ).to(device)
            else:
                model = TopTaggingCosLinear(
                    embedding_dim=args.embedding_dim,
                    particle_dim=args.particle_dim,
                    s=args.scale
                ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=2e-3,
                total_steps=total_scheduler_steps if total_scheduler_steps > 0 else 1
            )
            print(f"Params: {count_parameters(model):,}")

            best_val_acc = 0.0
            history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(args.epochs):
                if use_chunking:
                    chunk_indices = list(range(num_train_chunks))
                    np.random.shuffle(chunk_indices)
                else:
                    chunk_indices = [0]

                epoch_loss_sum = 0.0
                epoch_acc_sum = 0.0
                epoch_events = 0

                for c_idx, chunk_idx in enumerate(chunk_indices):
                    start = chunk_idx * args.chunk_size
                    n_events = min(args.chunk_size, train_len - start)

                    if use_chunking:
                        tr_ev, tr_lab, tr_w = load_awkward(
                            train_path, max_events=n_events, start_event=start, lazy=True
                        )
                    else:
                        tr_ev, tr_lab, tr_w = cached_train_data
                    tr_ds = JetTaggingDataset(tr_ev, tr_lab, tr_w)
                    tr_loader = DataLoader(
                        tr_ds, batch_size=args.batch_size, shuffle=True,
                        pin_memory=True, num_workers=0
                    )

                    c_loss, c_acc = train_epoch(
                        model, tr_loader, optimizer, device,
                        scheduler=scheduler,
                        accumulation_steps=args.accumulation_steps,
                        scaler=scaler
                    )

                    epoch_loss_sum += c_loss * n_events
                    epoch_acc_sum += c_acc * n_events
                    epoch_events += n_events

                    if use_chunking:
                        del tr_loader, tr_ds, tr_ev, tr_lab, tr_w
                        gc.collect()
                        empty_cache(backend_type)

                t_loss = epoch_loss_sum / epoch_events
                t_acc = epoch_acc_sum / epoch_events

                # Evaluate on validation set (use cached data in no-chunk mode)
                val_metrics = evaluate(
                    model, val_path, device, backend_type=backend_type,
                    chunk_size=args.chunk_size, cached_data=cached_val_data
                )

                history['train_loss'].append(t_loss)
                history['train_acc'].append(t_acc)
                history['val_acc'].append(val_metrics['accuracy'])

                is_best = val_metrics['accuracy'] > best_val_acc
                if is_best:
                    best_val_acc = val_metrics['accuracy']
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

                print(f"Epoch {epoch+1:2d}/{args.epochs}: "
                      f"Train Loss={t_loss:.4f} Acc={t_acc:.4f} | "
                      f"Val Acc={val_metrics['accuracy']:.4f}"
                      f"{' [NEW BEST]' if is_best else ''}")

            # Final evaluation on full test set (use cached data in no-chunk mode)
            test_metrics = evaluate(
                model, test_path, device, backend_type=backend_type,
                chunk_size=args.chunk_size, cached_data=cached_test_data
            )
            test_metrics['params'] = count_parameters(model)

            # Save results directly to output_dir (run_all.py includes model/seed in path)
            os.makedirs(args.output_dir, exist_ok=True)

            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(test_metrics, f, indent=2)
            with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

            print(f"\nTest: Acc={test_metrics['accuracy']:.4f} "
                  f"Prec={test_metrics['precision']:.4f} "
                  f"Rec={test_metrics['recall']:.4f} "
                  f"F1={test_metrics['f1_score']:.4f} "
                  f"AUC={test_metrics['roc_auc']:.4f}")
            print(f"Saved to {args.output_dir}")

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
