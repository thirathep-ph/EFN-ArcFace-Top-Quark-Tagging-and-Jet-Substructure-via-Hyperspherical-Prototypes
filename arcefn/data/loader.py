"""
Step 1: H5 -> Awkward Array -> Vector
======================================
Convert HDF5 dataset into awkward jagged arrays of
4-vectors for FastJet processing and EFN training.

Pipeline: H5 Dataset -> NumPy -> Awkward (jagged) -> Vector (Lorentz)
"""
import hdf5plugin
import h5py
import numpy as np
import awkward as ak
import vector

# Ensure vector is registered with awkward
vector.register_awkward()

N_CONS = 200  # max constituents per event


class LazyAwkwardEvents:
    """Lazy wrapper around raw NumPy 4-momentum arrays.

    Stores zero-padded (N, 200) arrays to delay Awkward conversion
    until __getitem__ is called. FastJet-incompatible; use lazy=False
    when calling load_awkward() if FastJet clustering is needed.
    """

    def __init__(self, es: np.ndarray, pxs: np.ndarray, pys: np.ndarray, pzs: np.ndarray, trim_padding: bool = True):
        """Store raw NumPy arrays, deferring Awkward conversion until access.

        Parameters
        ----------
        es : np.ndarray
            Energies of shape (N, 200).
        pxs : np.ndarray
            x-momenta of shape (N, 200).
        pys : np.ndarray
            y-momenta of shape (N, 200).
        pzs : np.ndarray
            z-momenta of shape (N, 200).
        trim_padding : bool
            Whether to remove zero-padded constituents on access (default True).
        """
        self.E = es
        self.px = pxs
        self.py = pys
        self.pz = pzs
        self.trim_padding = trim_padding

    def __len__(self):
        """Return number of events."""
        return len(self.E)

    def __getitem__(self, idx):
        """Return jagged Momentum4D array for given index or slice.

        Parameters
        ----------
        idx : int, np.integer, or slice
            Index or slice of events to retrieve.

        Returns
        -------
        ak.Array
            Jagged array of Momentum4D (LorentzVector) events.
        """
        E_sub = self.E[idx]
        px_sub = self.px[idx]
        py_sub = self.py[idx]
        pz_sub = self.pz[idx]

        is_single = isinstance(idx, (int, np.integer))
        if is_single:
            E_sub = np.expand_dims(E_sub, 0)
            px_sub = np.expand_dims(px_sub, 0)
            py_sub = np.expand_dims(py_sub, 0)
            pz_sub = np.expand_dims(pz_sub, 0)

        if self.trim_padding:
            mask = E_sub > 0
            n_per_event = mask.sum(axis=1)

            E_flat = E_sub[mask]
            px_flat = px_sub[mask]
            py_flat = py_sub[mask]
            pz_flat = pz_sub[mask]

            counts = n_per_event.tolist()
            e_jagged   = ak.unflatten(E_flat, counts)
            px_jagged  = ak.unflatten(px_flat, counts)
            py_jagged  = ak.unflatten(py_flat, counts)
            pz_jagged  = ak.unflatten(pz_flat, counts)
        else:
            e_jagged  = ak.from_numpy(E_sub)
            px_jagged = ak.from_numpy(px_sub)
            py_jagged = ak.from_numpy(py_sub)
            pz_jagged = ak.from_numpy(pz_sub)

        events_sliced = ak.zip({
            'E': e_jagged,
            'px': px_jagged,
            'py': py_jagged,
            'pz': pz_jagged,
        }, with_name="Momentum4D")

        if is_single:
            return events_sliced[0]
        return events_sliced


def load_awkward(
    path: str,
    max_events: int | None = None,
    start_event: int = 0,
    trim_padding: bool = True,
    lazy: bool = True,
) -> tuple:
    """Load H5 file into awkward arrays of particle 4-vectors.

    Parameters
    ----------
    path : str
        Path to H5 dataset file (Top Tagging Reference Dataset format).
    max_events : int or None, optional
        Maximum number of events to load. If None, loads all.
    start_event : int, optional
        Starting event index (default 0).
    trim_padding : bool, optional
        Remove zero-padded constituents (default True).
    lazy : bool, optional
        If True, return LazyAwkwardEvents wrapper (saves memory but
        incompatible with FastJet batch processing; default True).

    Returns
    -------
    events : ak.Array or LazyAwkwardEvents
        Jagged array of Momentum4D (LorentzVectors). If lazy=True,
        returns LazyAwkwardEvents; if lazy=False, returns genuine
        ak.Array compatible with FastJet batch clustering.
    labels : np.ndarray
        Binary labels (0 = QCD, 1 = Top).
    weights : np.ndarray
        Event weights (all ones if not provided in dataset).
    """
    f = h5py.File(path, 'r')
    
    # Locate dataset (handle table/table or direct keys)
    if 'table' in f:
        tbl = f['table']['table']
    else:
        tbl = f[list(f.keys())[0]]
        
    total_len = tbl.shape[0]
    start = start_event
    end = min(total_len, start + max_events) if max_events is not None else total_len
    n = end - start

    if n <= 0:
        f.close()
        raise ValueError(f"Invalid range: start_event={start_event}, max_events={max_events} for dataset of length {total_len}")

    print(f"  Reading {n:,} entries (range {start:,} to {end:,}) from {path}...")
    
    # Layout check: values_block_0 (804 cols) vs values_block_1 (2 cols)
    if 'values_block_0' in tbl.dtype.names:
        v0 = tbl['values_block_0'][start:end]
        v1 = tbl['values_block_1'][start:end]
        
        # Extract 4-momenta: interleaved (E, px, py, pz) per particle
        es  = v0[:, 0:N_CONS*4:4]
        pxs = v0[:, 1:N_CONS*4:4]
        pys = v0[:, 2:N_CONS*4:4]
        pzs = v0[:, 3:N_CONS*4:4]
        labels = v1[:, 1].astype(np.int8)
    else:
        # Fallback for alternative layouts (e.g. flat array)
        raw = tbl[start:end].astype(np.float32)
        es  = raw[:, 0:N_CONS*4:4]
        pxs = raw[:, 1:N_CONS*4:4]
        pys = raw[:, 2:N_CONS*4:4]
        pzs = raw[:, 3:N_CONS*4:4]
        if raw.shape[1] > 800:
            labels = raw[:, 800].astype(np.int8)
        else:
            raise ValueError(
                f"Flat H5 layout missing the label column (shape[1]={raw.shape[1]} <= 800); "
                "cannot proceed without labels.")

    f.close()
    weights = np.ones(n, dtype=np.float32)

    if lazy:
        events = LazyAwkwardEvents(es, pxs, pys, pzs, trim_padding=trim_padding)
        return events, labels, weights

    if trim_padding:
        # E=0 means zero-padded
        mask = es > 0
        n_per_event = mask.sum(axis=1)
        
        es_flat  = es[mask]
        pxs_flat = pxs[mask]
        pys_flat = pys[mask]
        pzs_flat = pzs[mask]

        counts = n_per_event.tolist()
        e_jagged   = ak.unflatten(es_flat, counts)
        px_jagged  = ak.unflatten(pxs_flat, counts)
        py_jagged  = ak.unflatten(pys_flat, counts)
        pz_jagged  = ak.unflatten(pzs_flat, counts)
    else:
        e_jagged  = ak.from_numpy(es)
        px_jagged = ak.from_numpy(pxs)
        py_jagged = ak.from_numpy(pys)
        pz_jagged = ak.from_numpy(pzs)

    # Build Lorentz vectors
    events = ak.zip({
        'E': e_jagged,
        'px': px_jagged,
        'py': py_jagged,
        'pz': pz_jagged,
    }, with_name="Momentum4D")

    return events, labels, weights


def get_h5_len(path: str) -> int:
    """Get number of events in an H5 dataset file.

    Parameters
    ----------
    path : str
        Path to H5 file.

    Returns
    -------
    int
        Number of events.
    """
    with h5py.File(path, 'r') as f:
        if 'table' in f:
            tbl = f['table']['table']
        else:
            tbl = f[list(f.keys())[0]]
        return tbl.shape[0]


def load_all_labels(path: str, max_events: int | None = None) -> np.ndarray:
    """Load only the label column for the first ``max_events`` events.

    Mirrors the layout logic of :func:`load_awkward` without touching the
    (heavy) 4-momentum blocks. Used by null-control retrains that need the
    full label vector in memory while streaming events in chunks.
    """
    with h5py.File(path, 'r') as f:
        if 'table' in f:
            tbl = f['table']['table']
        else:
            tbl = f[list(f.keys())[0]]
        total_len = tbl.shape[0]
        n = min(total_len, max_events) if max_events is not None else total_len
        if 'values_block_0' in tbl.dtype.names:
            return tbl['values_block_1'][:n, 1].astype(np.int8)
        raw = tbl[:n]
        if raw.shape[1] > 800:
            return raw[:, 800].astype(np.int8)
        raise ValueError(
            f"Flat H5 layout missing the label column (shape[1]={raw.shape[1]} <= 800).")


def load_events_by_indices(path, indices, trim_padding=True):
    """
    Load specific events by their global indices from the H5 file.
    Returns Momentum4D awkward array.
    """
    indices = np.array(indices, dtype=np.int64)
    sort_idx = np.argsort(indices)
    sorted_indices = indices[sort_idx]
    
    unique_indices, unique_inverse = np.unique(sorted_indices, return_inverse=True)
    
    with h5py.File(path, 'r') as f:
        if 'table' in f:
            tbl = f['table']['table']
        else:
            tbl = f[list(f.keys())[0]]
            
        if 'values_block_0' in tbl.dtype.names:
            v0 = tbl['values_block_0'][unique_indices]
        else:
            v0 = tbl[unique_indices].astype(np.float32)
            
    # Restore the original requested order (including duplicates)
    v0_sorted = v0[unique_inverse]
    # Invert sorting index
    inv_sort_idx = np.zeros_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(len(indices))
    v0_original = v0_sorted[inv_sort_idx]
    
    es  = v0_original[:, 0:N_CONS*4:4]
    pxs = v0_original[:, 1:N_CONS*4:4]
    pys = v0_original[:, 2:N_CONS*4:4]
    pzs = v0_original[:, 3:N_CONS*4:4]
        
    if trim_padding:
        mask = es > 0
        n_per_event = mask.sum(axis=1)
        
        es_flat  = es[mask]
        pxs_flat = pxs[mask]
        pys_flat = pys[mask]
        pzs_flat = pzs[mask]

        counts = n_per_event.tolist()
        e_jagged   = ak.unflatten(es_flat, counts)
        px_jagged  = ak.unflatten(pxs_flat, counts)
        py_jagged  = ak.unflatten(pys_flat, counts)
        pz_jagged  = ak.unflatten(pzs_flat, counts)
    else:
        e_jagged  = ak.from_numpy(es)
        px_jagged = ak.from_numpy(pxs)
        py_jagged = ak.from_numpy(pys)
        pz_jagged = ak.from_numpy(pzs)

    events = ak.zip({
        'E': e_jagged,
        'px': px_jagged,
        'py': py_jagged,
        'pz': pz_jagged,
    }, with_name="Momentum4D")

    return events


import torch
from torch.utils.data import Dataset

class JetTaggingDataset(Dataset):
    """PyTorch Dataset for the Top Tagging Reference Dataset.

    Wraps awkward events and provides padded tensors with padding masks
    for the EFN family of models.
    """

    def __init__(self, events, labels, weights):
        """Initialize dataset with events, labels, and weights.

        Pre-converts raw arrays to padded torch tensors (N, 200) for
        efficient indexing during training.

        Parameters
        ----------
        events : LazyAwkwardEvents or ak.Array
            Event 4-momentum data (either lazy NumPy-backed or awkward jagged).
        labels : np.ndarray
            Binary class labels of shape (N,).
        weights : np.ndarray
            Event weights of shape (N,).
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        if isinstance(events, LazyAwkwardEvents):
            self.events = events
            self.E = torch.from_numpy(events.E).float()
            self.px = torch.from_numpy(events.px).float()
            self.py = torch.from_numpy(events.py).float()
            self.pz = torch.from_numpy(events.pz).float()
        else:
            self.events = events
            # Convert awkward to padded numpy
            self.px = torch.tensor(ak.to_numpy(ak.fill_none(ak.pad_none(events.px, 200, clip=True), 0)), dtype=torch.float32)
            self.py = torch.tensor(ak.to_numpy(ak.fill_none(ak.pad_none(events.py, 200, clip=True), 0)), dtype=torch.float32)
            self.pz = torch.tensor(ak.to_numpy(ak.fill_none(ak.pad_none(events.pz, 200, clip=True), 0)), dtype=torch.float32)
            self.E = torch.tensor(ak.to_numpy(ak.fill_none(ak.pad_none(events.E, 200, clip=True), 0)), dtype=torch.float32)
        
    def __len__(self):
        """Return number of events in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return a single event as padded tensor with mask and metadata.

        Parameters
        ----------
        idx : int
            Event index.

        Returns
        -------
        x : torch.Tensor
            Constituent 4-momenta of shape (200, 4).
        label : torch.Tensor
            Class label (0 for QCD, 1 for Top).
        weight : torch.Tensor
            Event weight.
        mask : torch.Tensor
            Padding mask of shape (200,) where 1 = real constituent.
        idx : int
            Original event index (for stratified sampling, re-weighting).
        """
        x = torch.stack([self.E[idx], self.px[idx], self.py[idx], self.pz[idx]], dim=-1)
        mask = (x[:, 0] > 0).float()
        return x, self.labels[idx], self.weights[idx], mask, idx

def print_summary(events, labels):
    """Print a quick summary of event counts, constituents, and label balance.

    Parameters
    ----------
    events : LazyAwkwardEvents or ak.Array
        Event data.
    labels : np.ndarray
        Binary class labels.
    """
    n_events = len(events)
    if isinstance(events, LazyAwkwardEvents):
        n_per = (events.E > 0).sum(axis=1)
    else:
        n_per = ak.num(events)
    print(f"  Events: {n_events:,}")
    print(f"  Constituents: mean={np.mean(n_per):.1f}, max={np.max(n_per)}")
    n_top = int(labels.sum())
    print(f"  Labels: {n_top} Top, {n_events - n_top} QCD")


if __name__ == '__main__':
    for split in ['data/top_tagging/train.h5', 'data/top_tagging/val.h5', 'data/top_tagging/test.h5']:
        try:
            events, labels, weights = load_awkward(split, max_events=1000, trim_padding=True)
            print_summary(events, labels)
        except Exception as e:
            print(f"  Could not load {split}: {e}")
