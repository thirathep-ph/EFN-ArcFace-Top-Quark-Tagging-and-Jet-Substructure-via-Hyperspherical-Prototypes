import torch


def test_get_device():
    from arcefn.utils.device import get_device
    result = get_device()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.device)
    assert isinstance(result[1], str)


def test_get_device_backend():
    from arcefn.utils.device import get_device
    _, backend = get_device()
    assert backend in ('cuda', 'dml', 'cpu')


def test_count_parameters_smoke():
    from arcefn.models.efn_arcface import TopTaggingModel, count_parameters
    model = TopTaggingModel()
    n = count_parameters(model)
    assert n > 0
    assert isinstance(n, int)
