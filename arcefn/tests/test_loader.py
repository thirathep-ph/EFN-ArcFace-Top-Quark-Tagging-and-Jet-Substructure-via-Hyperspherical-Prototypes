import os
import numpy as np


def test_get_h5_len():
    from arcefn.data.loader import get_h5_len

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    n = get_h5_len(test_path)
    assert n == 404000, f"Expected 404000, got {n}"


def test_load_awkward_lazy():
    from arcefn.data.loader import load_awkward

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events, labels, weights = load_awkward(test_path, max_events=100, lazy=True)
    assert len(events) == 100
    assert len(labels) == 100
    assert len(weights) == 100


def test_load_awkward_nonlazy():
    from arcefn.data.loader import load_awkward

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events, labels, weights = load_awkward(test_path, max_events=100, lazy=False)
    assert len(events) == 100
    assert len(labels) == 100
    assert len(weights) == 100


def test_load_events_by_indices():
    from arcefn.data.loader import load_events_by_indices

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    indices = [0, 10, 100, 1000, 10000]
    events = load_events_by_indices(test_path, indices)
    assert len(events) == 5


def test_jet_tagging_dataset():
    from arcefn.data.loader import JetTaggingDataset, load_awkward
    from torch.utils.data import DataLoader

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events, labels, weights = load_awkward(test_path, max_events=100, lazy=False)
    dataset = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(dataset, batch_size=10)

    batch = next(iter(loader))
    constituents, labels, weights, mask, indices = batch
    assert constituents.shape[0] == 10
    assert constituents.shape[2] == 4  # (E, px, py, pz)
    assert labels.shape[0] == 10
    assert mask.shape[0] == 10
    assert mask.shape[1] == 200


import pytest
