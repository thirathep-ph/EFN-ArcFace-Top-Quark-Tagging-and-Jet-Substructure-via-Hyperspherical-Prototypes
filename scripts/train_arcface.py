"""
Training Pipeline for EFN + ArcFace Prototype Layer
===================================================
Training-only entry point.  Strips all post-training analysis (UMAP,
clustering, Lund planes, plots) — those belong in the ``analysis/``
scripts.

Usage
-----
::

    python scripts/train_arcface.py --epochs 50 --amp
    python scripts/train_arcface.py --quick
"""
import json
import os
import random
import time
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device, empty_cache, amp_available, get_amp_scaler
from arcefn.models.efn_arcface import TopTaggingModel, arcface_loss, count_parameters
from arcefn.data.loader import load_awkward, JetTaggingDataset, get_h5_len, load_all_labels


# ============================================================================
# Training & Evaluation Loops
# ============================================================================

def train_chunk(model, loader, optimizer, device, scheduler=None,
                accumulation_steps=8, scaler=None):
    """
    Train the model for one chunk (or full epoch) with gradient accumulation.

    Parameters
    ----------
    model : torch.nn.Module
        The EFN + ArcFace model.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Compute device.
    scheduler : torch.optim.lr_scheduler, optional
        LR scheduler (stepped every *accumulation_steps*).
    accumulation_steps : int
        Number of mini-batches to accumulate before stepping.
    scaler : torch.cuda.amp.GradScaler, optional
        AMP scaler for FP16 training.

    Returns
    -------
    avg_loss : float
        Average loss over the chunk.
    avg_acc : float
        Average accuracy over the chunk.
    """
    model.train()
    total_loss, total_acc = 0, 0
    criterion_cls = arcface_loss

    optimizer.zero_grad()
    n_batches = len(loader)

    for batch_idx, (x, y, w, m, _) in enumerate(loader):
        x, y, w, m = x.to(device), y.to(device), w.to(device), m.to(device)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, sims, _ = model(x, labels=y, mask=m)
                loss = criterion_cls(logits, y)
        else:
            logits, sims, _ = model(x, labels=y, mask=m)
            loss = criterion_cls(logits, y)
        loss = loss / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    try:
                        scheduler.step()
                    except ValueError:
                        pass
        else:
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    try:
                        scheduler.step()
                    except ValueError:
                        pass

        total_loss += loss.item() * accumulation_steps
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, file_path, device, backend_type='cpu', batch_size=1024,
             chunk_size=100000, max_events=None, cached_data=None):
    """
    Evaluate the model on a dataset split (chunked or single-load).

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    file_path : str or Path
        Path to ``*.h5`` dataset.
    device : torch.device
        Compute device.
    backend_type : str
        Backend identifier (``'cpu'``, ``'cuda'``, ``'dml'``).
    batch_size : int
        Batch size for evaluation.
    chunk_size : int
        Events per chunk (ignored when *cached_data* is provided).
    max_events : int, optional
        Maximum events to load.
    cached_data : tuple, optional
        Pre-loaded ``(events, labels, weights)`` for single-load mode.

    Returns
    -------
    avg_loss : float
    avg_acc : float
    """
    model.eval()

    if cached_data is not None:
        ev, lab, w = cached_data
        ds = JetTaggingDataset(ev, lab, w)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        total_len = len(ds)
        num_chunks = 1
    else:
        total_len = get_h5_len(file_path)
        if max_events is not None:
            total_len = min(total_len, max_events)
        num_chunks = int(np.ceil(total_len / chunk_size))

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    criterion_cls = arcface_loss

    with torch.no_grad():
        if cached_data is not None:
            for x, y, w_b, m, _ in loader:
                x, y, w_b, m = x.to(device), y.to(device), w_b.to(device), m.to(device)
                logits, sims, _ = model(x, mask=m)
                loss = criterion_cls(logits, y)
                total_loss += loss.item()
                total_acc += (logits.argmax(dim=1) == y).float().mean().item()
                total_batches += 1
        else:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                n_events = min(chunk_size, total_len - start)

                ev, lab, w = load_awkward(file_path, max_events=n_events,
                                          start_event=start, lazy=True)
                ds = JetTaggingDataset(ev, lab, w)
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

                for x, y, w_b, m, _ in loader:
                    x, y, w_b, m = x.to(device), y.to(device), w_b.to(device), m.to(device)
                    logits, sims, _ = model(x, mask=m)
                    loss = criterion_cls(logits, y)
                    total_loss += loss.item()
                    total_acc += (logits.argmax(dim=1) == y).float().mean().item()
                    total_batches += 1

                del loader, ds, ev, lab, w
                gc.collect()
                empty_cache(backend_type)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0
    return avg_loss, avg_acc


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Parse arguments, initialise model / data / training loop, and save the best
    checkpoint and training history.
    """
    parser = argparse.ArgumentParser(
        description="Train EFN + ArcFace (no post-training analysis)")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_events', type=int, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--scale', type=float, default=16.0,
                        help='ArcFace scale parameter s')
    parser.add_argument('--margin', type=float, default=None,
                        help='ArcFace margin m (default: 0.5, warmup from 0.0)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='EFN embedding/latent dimension')
    parser.add_argument('--particle_dim', type=int, default=128,
                        help='EFN particle/phi hidden dimension')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--no-chunk', action='store_true',
                        help='Load entire dataset into RAM once (faster)')
    parser.add_argument('--chunk_size', type=int, default=100000)
    parser.add_argument('--train-only', action='store_true',
                        help='Alias for compatibility; no effect (analysis stripped)')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision (FP16)')
    parser.add_argument('--shuffle_labels', type=int, default=None,
                        help='Null-control: globally permute TRAIN labels once '
                             'with this seed (validation labels stay true).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device, backend_type = get_device()
    use_amp = args.amp and amp_available(backend_type)
    scaler = get_amp_scaler(backend_type, enabled=use_amp)

    max_events = args.max_events if args.max_events else (5000 if args.quick else None)

    train_path = DATA_DIR / 'train.h5'
    val_path = DATA_DIR / 'val.h5'

    train_len = get_h5_len(train_path)
    val_len = get_h5_len(val_path)
    if max_events:
        train_len = min(train_len, max_events)
        val_len = min(val_len, max_events)

    use_chunking = not args.no_chunk

    label_perm = None
    full_lab = None
    if args.shuffle_labels is not None:
        rng = np.random.default_rng(args.shuffle_labels)
        label_perm = rng.permutation(train_len)
        full_lab = load_all_labels(train_path, train_len)
        assert len(full_lab) == train_len == len(label_perm)
        print(f"Null-control: train labels globally permuted "
              f"(seed {args.shuffle_labels}); validation labels untouched.")

    if use_chunking:
        num_train_chunks = int(np.ceil(train_len / args.chunk_size))
        total_train_batches = 0
        for chunk_idx in range(num_train_chunks):
            start = chunk_idx * args.chunk_size
            n_events = min(args.chunk_size, train_len - start)
            total_train_batches += int(np.ceil(n_events / args.batch_size))
    else:
        num_train_chunks = 1
        total_train_batches = int(np.ceil(train_len / args.batch_size))

    steps_per_epoch = int(np.ceil(total_train_batches / args.accumulation_steps))
    total_scheduler_steps = steps_per_epoch * args.epochs + 500  # +500 buffer: covers chunk remainder (1187 vs 1184) and any ceil drift; scheduler is try-guarded below

    print("Dataset configuration:")
    print(f"  Train events: {train_len:,}")
    print(f"  Val events:   {val_len:,}")
    print(f"  Total batches per epoch: {total_train_batches:,}")

    model = TopTaggingModel(
        s=args.scale, m=0.0,
        embedding_dim=args.embedding_dim,
        particle_dim=args.particle_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3,
        total_steps=total_scheduler_steps if total_scheduler_steps > 0 else 1,
    )

    warmup_epochs = 10
    final_margin = args.margin if args.margin is not None else 0.5
    print(f"\nTraining EFN + ArcFace (Params: {count_parameters(model):,})")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    cached_train_data = None
    cached_val_data = None
    if not use_chunking:
        print("Pre-loading all data into memory (single-load mode)...")
        tr_ev, tr_lab, tr_w = load_awkward(train_path, max_events=train_len, lazy=True)
        if label_perm is not None:
            tr_lab = np.asarray(tr_lab)[label_perm]
        cached_train_data = (tr_ev, tr_lab, tr_w)
        vl_ev, vl_lab, vl_w = load_awkward(val_path, max_events=val_len, lazy=True)
        cached_val_data = (vl_ev, vl_lab, vl_w)

    epoch_pbar = tqdm(range(args.epochs), desc="Overall Progress")
    for epoch in epoch_pbar:
        current_m = final_margin * min(1.0, (epoch + 1) / warmup_epochs)
        model.arcface_head.set_margin(current_m)

        if use_chunking:
            chunk_indices = list(range(num_train_chunks))
            np.random.shuffle(chunk_indices)
        else:
            chunk_indices = [0]

        t_loss_sum, t_acc_sum = 0.0, 0.0
        events_trained = 0

        desc = f"Epoch {epoch + 1}/{args.epochs}"
        if use_chunking:
            desc += f" [Chunk 0/{num_train_chunks}]"
        epoch_pbar_inner = tqdm(total=total_train_batches, desc=desc, leave=False)

        for c_idx, chunk_idx in enumerate(chunk_indices):
            if use_chunking:
                epoch_pbar_inner.set_description(
                    f"Epoch {epoch + 1}/{args.epochs} [Chunk {c_idx + 1}/{num_train_chunks}]")
                start = chunk_idx * args.chunk_size
                n_events = min(args.chunk_size, train_len - start)
                tr_ev, tr_lab, tr_w = load_awkward(
                    train_path, max_events=n_events, start_event=start, lazy=True)
                if label_perm is not None:
                    tr_lab = full_lab[label_perm[start:start + n_events]]
            else:
                n_events = train_len
                tr_ev, tr_lab, tr_w = cached_train_data

            train_ds = JetTaggingDataset(tr_ev, tr_lab, tr_w)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                pin_memory=True, num_workers=0)

            c_loss, c_acc = train_chunk(
                model, train_loader, optimizer, device, scheduler=scheduler,
                accumulation_steps=args.accumulation_steps, scaler=scaler)

            t_loss_sum += c_loss * n_events
            t_acc_sum += c_acc * n_events
            events_trained += n_events

            epoch_pbar_inner.update(len(train_loader))
            epoch_pbar_inner.set_postfix(Loss=f"{c_loss:.4f}", Acc=f"{c_acc:.4f}")

            if use_chunking:
                del train_loader, train_ds, tr_ev, tr_lab, tr_w
                gc.collect()
                empty_cache(backend_type)

        epoch_pbar_inner.close()

        t_loss = t_loss_sum / events_trained
        t_acc = t_acc_sum / events_trained

        v_loss, v_acc = evaluate(
            model, val_path, device, backend_type=backend_type,
            batch_size=1024, chunk_size=args.chunk_size,
            max_events=val_len, cached_data=cached_val_data)

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        curr_lr = optimizer.param_groups[0]['lr']
        tag = " [NEW BEST]" if v_acc > best_acc else ""
        if v_acc > best_acc:
            best_acc = v_acc
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

        print(f"Epoch {epoch + 1:2d}/{args.epochs}: m={current_m:.2f} "
              f"| LR={curr_lr:.6f}")
        print(f"  Train: Loss={t_loss:.4f}, Acc={t_acc:.4f} "
              f"| Val: Acc={v_acc:.4f}{tag}")

    # Save training history
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Save a run-config snapshot alongside the checkpoint for reproducibility
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)

    print(f"\n[COMPLETE] Best val acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()
