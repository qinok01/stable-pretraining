"""Focused tests for TS2Vec encoder, hierarchical loss, and temporal crops."""

import pytest
import torch
import torch.nn.functional as F

from stable_pretraining.backbone.ts2vec import TS2VecEncoder
from stable_pretraining.forward import _ts2vec_crops
from stable_pretraining.losses.ts2vec import TS2VecLoss
from stable_pretraining.methods.ts2vec import TS2Vec


def _dense_pair_loss(z1: torch.Tensor, z2: torch.Tensor, dim: int) -> torch.Tensor:
    """Reference pair loss using explicit candidate indexing and cross entropy."""
    first = torch.cat((z1, z2), dim=dim)
    second = first.transpose(dim, dim + 1)
    similarity = torch.matmul(first, second)
    size = z1.shape[dim]
    # Move the contrast axis to the end, then form one row per anchor.
    similarity = similarity.movedim(dim, -2)
    rows = []
    targets = []
    for anchor in range(2 * size):
        keep = torch.arange(2 * size, device=z1.device) != anchor
        positive = (anchor + size) % (2 * size)
        candidate_indices = torch.arange(2 * size, device=z1.device)[keep]
        rows.append(similarity[..., anchor, :][..., keep])
        targets.append((candidate_indices == positive).nonzero().item())
    logits = torch.stack(rows, dim=-2)
    target = torch.tensor(targets, dtype=torch.long, device=z1.device)
    return F.cross_entropy(
        logits.reshape(-1, 2 * size - 1), target.repeat(logits.shape[0])
    )


def _dense_hierarchical_reference(
    z1: torch.Tensor, z2: torch.Tensor, alpha: float, temporal_unit: int
) -> torch.Tensor:
    """Dense hierarchical objective, independently excluding each self-logit."""
    total = z1.sum() * 0
    levels = 0
    depth = 0
    while z1.size(1) > 1:
        if alpha:
            # At each timestamp, the batch is the contrast axis.
            total = total + alpha * _dense_pair_loss(
                z1.transpose(0, 1), z2.transpose(0, 1), 1
            )
        if depth >= temporal_unit and alpha < 1:
            # At each series, time is the contrast axis.
            total = total + (1 - alpha) * _dense_pair_loss(z1, z2, 1)
        z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
        depth += 1
        levels += 1
    if z1.size(1) == 1:
        if alpha:
            total = total + alpha * _dense_pair_loss(
                z1.transpose(0, 1), z2.transpose(0, 1), 1
            )
        levels += 1
    return total / levels


@pytest.mark.unit
def test_ts2vec_loss_matches_dense_reference_values_and_gradients():
    torch.manual_seed(23)
    actual_a = torch.randn(3, 7, 4, dtype=torch.float64, requires_grad=True)
    actual_b = torch.randn(3, 7, 4, dtype=torch.float64, requires_grad=True)
    expected_a = actual_a.detach().clone().requires_grad_()
    expected_b = actual_b.detach().clone().requires_grad_()

    actual = TS2VecLoss(alpha=0.37, temporal_unit=1)(actual_a, actual_b)
    expected = _dense_hierarchical_reference(expected_a, expected_b, 0.37, 1)
    actual_grads = torch.autograd.grad(actual, (actual_a, actual_b))
    expected_grads = torch.autograd.grad(expected, (expected_a, expected_b))

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("temporal_unit", [0, 2])
def test_ts2vec_loss_alpha_endpoints_and_temporal_unit(alpha, temporal_unit):
    torch.manual_seed(7)
    z1 = torch.randn(2, 9, 3)
    z2 = torch.randn(2, 9, 3)
    actual = TS2VecLoss(alpha=alpha, temporal_unit=temporal_unit)(z1, z2)
    expected = _dense_hierarchical_reference(z1, z2, alpha, temporal_unit)
    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_ts2vec_loss_single_sample_and_single_time_are_finite():
    # Either contrastive axis alone remains meaningful; only both degenerate
    # together produce the all-zero objective.
    for batch, time in ((1, 5), (3, 1)):
        z1 = torch.randn(batch, time, 2)
        z2 = torch.randn(batch, time, 2)
        assert torch.isfinite(TS2VecLoss()(z1, z2))


@pytest.mark.unit
def test_ts2vec_loss_both_axes_one_is_differentiable_zero():
    batch, time = 1, 1
    z1 = torch.randn(batch, time, 2, requires_grad=True)
    z2 = torch.randn(batch, time, 2, requires_grad=True)
    loss = TS2VecLoss()(z1, z2)
    assert loss.ndim == 0
    assert loss.item() == 0
    grad1, grad2 = torch.autograd.grad(loss, (z1, z2))
    assert torch.count_nonzero(grad1) == 0
    assert torch.count_nonzero(grad2) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"temporal_unit": -1}, "temporal_unit"),
    ],
)
def test_ts2vec_loss_rejects_invalid_configuration(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        TS2VecLoss(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "z1,z2",
    [
        (torch.randn(4, 5), torch.randn(4, 5)),
        (torch.randn(2, 5, 3), torch.randn(3, 5, 3)),
        (torch.randn(2, 5, 3), torch.randn(2, 4, 3)),
        (torch.randn(2, 5, 3), torch.randn(2, 5, 4)),
    ],
)
def test_ts2vec_loss_rejects_invalid_tensor_shapes(z1, z2):
    with pytest.raises((TypeError, ValueError)):
        TS2VecLoss()(z1, z2)


@pytest.mark.unit
def test_ts2vec_encoder_shapes_nan_handling_and_input_immutability():
    encoder = TS2VecEncoder(input_dims=3, output_dims=8, hidden_dims=4, depth=2).eval()
    x = torch.randn(2, 7, 3)
    x[0, 2, 1] = float("nan")
    original = x.clone()

    y = encoder(x)

    assert y.shape == (2, 7, 8)
    assert torch.isfinite(y).all()
    assert torch.equal(torch.isnan(x), torch.isnan(original))
    assert torch.equal(torch.nan_to_num(x), torch.nan_to_num(original))


@pytest.mark.unit
def test_ts2vec_encoder_eval_is_deterministic_and_supports_mask_modes():
    torch.manual_seed(11)
    encoder = TS2VecEncoder(input_dims=2, output_dims=6, hidden_dims=4, depth=2).eval()
    x = torch.randn(2, 5, 2)
    first = encoder(x)
    second = encoder(x)
    torch.testing.assert_close(first, second)

    all_false = encoder(x, mask="all_false")
    boolean_false = encoder(x, mask=torch.zeros(2, 5, dtype=torch.bool))
    all_true = encoder(x, mask="all_true")
    assert all_false.shape == all_true.shape == first.shape
    torch.testing.assert_close(all_false, boolean_false)
    assert not torch.equal(all_false, all_true)
    for mask in ("binomial", "all_true", "all_false"):
        result = encoder(x, mask=mask)
        assert result.shape == first.shape
        assert torch.isfinite(result).all()


@pytest.mark.unit
@pytest.mark.parametrize("x", [torch.randn(5, 3), torch.randn(2, 5, 4)])
def test_ts2vec_encoder_rejects_invalid_input_shape(x):
    encoder = TS2VecEncoder(input_dims=3, output_dims=4, hidden_dims=4, depth=1)
    with pytest.raises((TypeError, ValueError)):
        encoder(x)


@pytest.mark.unit
def test_ts2vec_encoder_rejects_unknown_mask():
    encoder = TS2VecEncoder(input_dims=2, output_dims=4, hidden_dims=4, depth=1)
    with pytest.raises((TypeError, ValueError), match="mask"):
        encoder(torch.randn(2, 5, 2), mask="unknown")


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ts2vec_crop_views_preserve_their_shared_temporal_alignment(seed):
    torch.manual_seed(seed)
    # Each time value identifies its original location, making alignment explicit.
    x = torch.arange(4 * 13 * 2, dtype=torch.float32).reshape(4, 13, 2)
    view1, view2, overlap = _ts2vec_crops(x, temporal_unit=1)

    assert 0 < overlap <= min(view1.size(1), view2.size(1))
    torch.testing.assert_close(view1[:, -overlap:], view2[:, :overlap])
    assert view1.shape[0] == view2.shape[0] == x.shape[0]
    assert view1.shape[2] == view2.shape[2] == x.shape[2]
    torch.testing.assert_close(
        x, torch.arange(4 * 13 * 2, dtype=torch.float32).reshape(4, 13, 2)
    )


@pytest.mark.unit
def test_ts2vec_method_returns_timestamp_and_pooled_embeddings_and_train_loss():
    torch.manual_seed(31)
    model = TS2Vec(
        input_dims=2,
        output_dims=6,
        hidden_dims=4,
        depth=1,
        alpha=0.5,
        temporal_unit=0,
    )
    batch = {"series": torch.randn(2, 12, 2)}

    model.eval()
    eval_result = model(batch, "val")
    assert eval_result["timestamp_embedding"].shape == (2, 12, 6)
    assert eval_result["embedding"].shape == (2, 6)
    assert "loss" not in eval_result

    model.train()
    train_result = model({"series": batch["series"].clone()}, "train")
    assert train_result["timestamp_embedding"].ndim == 3
    assert train_result["embedding"].shape == (2, 6)
    assert train_result["loss"].ndim == 0
    assert torch.isfinite(train_result["loss"])
    train_result["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
