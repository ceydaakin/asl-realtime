import pytest
import torch

from asl_realtime.landmarks import INPUT_SIZE, NUM_CLASSES
from asl_realtime.models import MODELS, build_model

N_FEATURES = 132


def _inputs(batch=3):
    torch.manual_seed(0)
    x = torch.randn(batch, INPUT_SIZE, N_FEATURES)
    mask = torch.ones(batch, INPUT_SIZE, dtype=torch.bool)
    mask[:, -10:] = False
    return x, mask


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_outputs_logits_per_class(name):
    model = build_model(name, n_features=N_FEATURES).eval()
    x, mask = _inputs()

    logits = model(x, mask)

    assert logits.shape == (3, NUM_CLASSES)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_ignores_values_in_padded_frames(name):
    model = build_model(name, n_features=N_FEATURES).eval()
    x, mask = _inputs()
    x_noisy = x.clone()
    x_noisy[:, -10:] = 1e3

    with torch.no_grad():
        a, b = model(x, mask), model(x_noisy, mask)

    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_prediction_does_not_depend_on_amount_of_padding(name):
    model = build_model(name, n_features=N_FEATURES).eval()
    x, mask = _inputs()
    extra = 32
    x_long = torch.cat([x, torch.randn(3, extra, N_FEATURES)], dim=1)
    mask_long = torch.cat([mask, torch.zeros(3, extra, dtype=torch.bool)], dim=1)

    real = int(mask[0].sum())

    with torch.no_grad():
        trimmed = model(x[:, :real], mask[:, :real])  # no padding at all, like a full live window
        padded = model(x, mask)
        long = model(x_long, mask_long)

    torch.testing.assert_close(padded, trimmed, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(long, trimmed, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_exposes_hparams_to_rebuild_it(name):
    model = build_model(name, n_features=N_FEATURES, hidden=64)

    rebuilt = build_model(name, n_features=N_FEATURES, **model.hparams)

    assert model.hparams["hidden"] == 64
    assert sum(p.numel() for p in rebuilt.parameters()) == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_trains_one_step(name):
    model = build_model(name, n_features=N_FEATURES)
    x, mask = _inputs()
    y = torch.tensor([0, 1, 2])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = torch.nn.functional.cross_entropy(model(x, mask), y)
    loss.backward()
    opt.step()

    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet", n_features=N_FEATURES)
