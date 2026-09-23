"""
Training Pipeline for LorentzNet + ArcFace Prototype Layer
===========================================================
Lorentz-equivariant graph backbone (Gong et al., JHEP 07 (2022) 030,
arXiv:2201.08187) with the SAME ArcFace hyperspherical prototype head as the
EFN pipeline, so downstream analysis (spectral clustering, eigengap, DGLAP
fits, prototype comparison) is directly comparable with the EFN results.

Training-only entry point (analysis lives in ``scripts/analysis/``).

Usage
-----
::

    python scripts/train_lorentznet.py --epochs 35 --batch_size 32
    python scripts/train_lorentznet.py --quick

NOTE (this machine): torch-directml only (AMD RX 5700 XT).  No --amp.
The per-pair (B,N,N,*) tensors dominate memory; use --max_constits to cap
the padded constituent dimension and --batch_size 32 (official) on 8GB.
"""
import json
import os
import random
import gc
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arcefn.utils.paths import DATA_DIR
from arcefn.utils.device import get_device, empty_cache
from arcefn.models.lorentznet_arcface import LorentzNetArcFace
from arcefn.models.efn_arcface import arcface_loss, count_parameters
from arcefn.data.loader import load_awkward, JetTaggingDataset, get_h5_len


# ============================================================================
# Training & Evaluation Loops (generic over model interface)
# ============================================================================

def train_chunk(model, loader, optimizer, device, scheduler=None,
                accumulation_steps=8, max_constits=None):
    """Train the model for one epoch with gradient accumulation.

    Parameters
    ----------
    model : torch.nn.Module
        Any model with ``forward(constituents, labels, mask) -> logits``.
    loader : DataLoader
        Training data loader (yields (x, y, w, m, idx)).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Compute device.
    scheduler : torch.optim.lr_scheduler, optional
        LR scheduler (stepped every *accumulation_steps*).
    accumulation_steps : int
        Mini-batches per optimizer step.
    max_constits : int, optional
        Cap on the padded constituent dimension (memory control).

    Returns
    -------
    avg_loss, avg_acc : float, float
    """
    model.train()
    total_loss, total_acc = 0, 0
    n_batches = len(loader)

    optimizer.zero_grad()
    for batch_idx, (x, y, w, m, _) in enumerate(loader):
        x, y, w, m = x.to(device), y.to(device), w.to(device), m.to(device)
        if max_constits is not None:
            x = x[:, :max_constits].contiguous()
            m = m[:, :max_constits].contiguous()

        logits, _, _ = model(x, labels=y, mask=m)
        loss = arcface_loss(logits, y) / accumulation_steps

        loss.backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * accumulation_steps
        total_acc += (logits.argmax(dim=1) == y).float().mean().item()

    return total_loss / n_batches, total_acc / n_batches


def evaluate(model, file_path, device, backend_type='cpu', batch_size=32,
             chunk_size=100000, max_events=None, max_constits=None):
    """Evaluate the model on a dataset split (chunked to bound memory)."""
    model.eval()
    total_len = get_h5_len(file_path)
    if max_events is not None:
        total_len = min(total_len, max_events)
    num_chunks = int(np.ceil(total_len / chunk_size))

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            n_events = min(chunk_size, total_len - start)
            ev, lab, w = load_awkward(file_path, max_events=n_events,
                                      start_event=start, lazy=True)
            ds = JetTaggingDataset(ev, lab, w)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            for x, y, w_b, m, _ in loader:
                x, y, w_b, m = x.to(device), y.to(device), w_b.to(device), m.to(device)
                if max_constits is not None:
                    x = x[:, :max_constits].contiguous()
                    m = m[:, :max_constits].contiguous()
                logits, _, _ = model(x, mask=m)
                total_loss += arcface_loss(logits, y).item()
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
    parser = argparse.ArgumentParser(
        description="Train LorentzNet + ArcFace (no post-training analysis)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='ArcFace margin warmup schedule length in epochs')
    parser.add_argument('--epoch_cooldown', type=float, default=0.0,
                        help='seconds to sleep after each epoch (GPU thermal '
                             'safety on the RX 5700 XT under DirectML)')
    parser.add_argument('--max_events', type=int, default=None)
    parser.add_argument('--val_events', type=int, default=None,
                        help='Validation events (default: same as --max_events)')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--scale', type=float, default=16.0,
                        help='ArcFace scale parameter s')
    parser.add_argument('--margin', type=float, default=None,
                        help='ArcFace margin m (default: 0.5, warmup from 0.0)')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--n_hidden', type=int, default=72,
                        help='LGEB hidden width (official LorentzNet: 72)')
    parser.add_argument('--n_layers', type=int, default=6,
                        help='Number of LGEB blocks (official: 6)')
    parser.add_argument('--c_weight', type=float, default=1e-3,
                        help='Coordinate-update scale c')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_constits', type=int, default=None,
                        help='Cap on padded constituents (memory control)')
    parser.add_argument('--output_dir', type=str, default='experiments/lorentznet_arcface')
    parser.add_argument('--checkpoint_dir', type=str, default='experiments/lorentznet_arcface')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--no-chunk', action='store_true',
                        help='Load entire dataset into RAM once (faster)')
    parser.add_argument('--chunk_size', type=int, default=100000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device, backend_type = get_device()

    max_events = args.max_events if args.max_events else (2000 if args.quick else None)

    train_path = DATA_DIR / 'train.h5'
    val_path = DATA_DIR / 'val.h5'

    train_len = get_h5_len(train_path)
    val_len = get_h5_len(val_path)
    if max_events:
        train_len = min(train_len, max_events)
    if args.val_events:
        val_len = min(val_len, args.val_events)
    elif max_events:
        val_len = min(val_len, max_events)

    use_chunking = not args.no_chunk
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
    total_scheduler_steps = steps_per_epoch * args.epochs

    print("Dataset configuration:")
    print(f"  Train events: {train_len:,}")
    print(f"  Val events:   {val_len:,}")
    print(f"  Total batches per epoch: {total_train_batches:,}")
    print(f"  Max constituents: {'uncapped' if args.max_constits is None else args.max_constits}")

    model = LorentzNetArcFace(
        s=args.scale, m=0.0,
        embedding_dim=args.embedding_dim,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        c_weight=args.c_weight,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3,
        total_steps=total_scheduler_steps if total_scheduler_steps > 0 else 1,
    )

    warmup_epochs = args.warmup_epochs
    final_margin = args.margin if args.margin is not None else 0.5
    print(f"\nTraining LorentzNet + ArcFace (Params: {count_parameters(model):,})")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    cached_train_data = None
    cached_val_data = None
    if not use_chunking:
        print("Pre-loading all data into memory (single-load mode)...")
        tr_ev, tr_lab, tr_w = load_awkward(train_path, max_events=train_len, lazy=True)
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
        epoch_pbar_inner = tqdm(total=total_train_batches, leave=False)

        for c_idx, chunk_idx in enumerate(chunk_indices):
            if use_chunking:
                start = chunk_idx * args.chunk_size
                n_events = min(args.chunk_size, train_len - start)
                tr_ev, tr_lab, tr_w = load_awkward(
                    train_path, max_events=n_events, start_event=start, lazy=True)
            else:
                n_events = train_len
                tr_ev, tr_lab, tr_w = cached_train_data

            train_ds = JetTaggingDataset(tr_ev, tr_lab, tr_w)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                pin_memory=True, num_workers=0)

            c_loss, c_acc = train_chunk(
                model, train_loader, optimizer, device, scheduler=scheduler,
                accumulation_steps=args.accumulation_steps,
                max_constits=args.max_constits)

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
            batch_size=32, chunk_size=args.chunk_size,
            max_events=val_len, max_constits=args.max_constits)

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

        if args.epoch_cooldown > 0 and epoch < args.epochs - 1:
            print(f"  cooldown {args.epoch_cooldown}s...")
            time.sleep(args.epoch_cooldown)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)

    print(f"\n[COMPLETE] Best val acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()