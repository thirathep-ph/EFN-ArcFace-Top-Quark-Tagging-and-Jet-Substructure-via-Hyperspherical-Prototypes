import os
import numpy as np
import pytest


def test_compute_features_small_batch():
    from arcefn.data.loader import load_events_by_indices
    from arcefn.utils.physics import compute_features

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events = load_events_by_indices(test_path, list(range(10)))
    feats = compute_features(events)

    assert feats.shape == (10, 10), f"Expected (10,10), got {feats.shape}"
    assert np.all(feats[:, 0] > 0), "Mass should be positive"
    assert np.all(feats[:, 8] >= 0) and np.all(feats[:, 8] <= 0.5)


def test_physics_feature_count():
    from arcefn.data.loader import load_events_by_indices
    from arcefn.utils.physics import compute_features

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events = load_events_by_indices(test_path, [0])
    feats = compute_features(events)
    expected = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    assert feats.shape[1] == len(expected)


def test_get_lund_coordinates():
    from arcefn.data.loader import load_events_by_indices
    from arcefn.utils.physics import get_lund_coordinates

    test_path = os.path.join('data', 'top_tagging', 'test.h5')
    if not os.path.exists(test_path):
        pytest.skip("Test H5 file not found")

    events = load_events_by_indices(test_path, list(range(5)))
    lund = get_lund_coordinates(events)
    assert lund is not None
