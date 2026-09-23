import numpy as np
import torch


def _dummy_batch(batch_size=4, n_const=50):
    data = np.random.randn(batch_size, 200, 4).astype(np.float32)
    data[:, :, 0] = np.abs(data[:, :, 0]) + 0.1
    for i in range(batch_size):
        data[i, n_const:, :] = 0.0
    return torch.from_numpy(data)


def test_efn_arcface_forward():
    from arcefn.models.efn_arcface import TopTaggingModel
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64)
    model.eval()
    x = _dummy_batch(4, 50)
    with torch.no_grad():
        logits, sims, emb = model(x)
    assert logits.shape == (4, 2)
    assert sims.shape == (4, 2)
    assert emb.shape == (4, 64)


def test_efn_linear_forward():
    from arcefn.models.efn_linear import TopTaggingLinear
    model = TopTaggingLinear(embedding_dim=64)
    model.eval()
    x = _dummy_batch(4, 50)
    with torch.no_grad():
        logits, emb = model(x)
    assert logits.shape == (4, 2)
    assert emb.shape == (4, 64)


def test_efn_coslinear_forward():
    from arcefn.models.efn_coslinear import TopTaggingCosLinear
    model = TopTaggingCosLinear(embedding_dim=64)
    model.eval()
    x = _dummy_batch(4, 50)
    with torch.no_grad():
        logits, emb = model(x)
    assert logits.shape == (4, 2)
    assert emb.shape == (4, 64)


def test_arcface_logits():
    from arcefn.models.efn_arcface import TopTaggingModel
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64)
    model.eval()
    x = _dummy_batch(8, 100)
    with torch.no_grad():
        logits, _, _ = model(x)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_linear_head():
    from arcefn.models.efn_linear import TopTaggingLinear
    model = TopTaggingLinear(embedding_dim=64)
    w = model.classifier.weight.detach().numpy()
    norms = np.linalg.norm(w, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-5), "Linear head not normalized"


def test_count_parameters():
    from arcefn.models.efn_arcface import TopTaggingModel, count_parameters
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64)
    n = count_parameters(model)
    assert n == 59072, f"Expected 59072, got {n}"


def test_efn_phi_output_dim():
    from arcefn.models.efn_arcface import TopTaggingModel
    model = TopTaggingModel(embedding_dim=64)
    assert model.efn.phi[6].out_features == 128


def test_efn_rho_output_dim():
    from arcefn.models.efn_arcface import TopTaggingModel
    model = TopTaggingModel(embedding_dim=64)
    assert model.efn.rho[3].out_features == 64


def test_arcface_centers_initialized():
    from arcefn.models.efn_arcface import TopTaggingModel
    model = TopTaggingModel()
    centers = model.arcface_head.class_centers.detach().numpy()
    assert centers.shape == (2, 64)
