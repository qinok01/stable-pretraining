"""Lifecycle tests for TS2Vec cumulative encoder averaging."""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from stable_pretraining._config import get_config
from stable_pretraining.callbacks.factories import _DEFAULT_CALLBACK_REGISTRY
from stable_pretraining.callbacks.ts2vec import TS2VecAveraging
from stable_pretraining.losses.ts2vec import TS2VecLoss
from stable_pretraining.methods.ts2vec import TS2Vec


def _model() -> TS2Vec:
    """Build a small deterministic TS2Vec model for lifecycle checks."""
    return TS2Vec(input_dims=2, output_dims=3, hidden_dims=4, depth=1)


def _set_gradients(module: torch.nn.Module, value: float) -> None:
    """Assign identical finite gradients to all parameters."""
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, value)


def _assert_state_is_average(
    actual: dict[str, torch.Tensor], snapshots: list[dict[str, torch.Tensor]]
) -> None:
    """Check that every averaged tensor includes each supplied snapshot once."""
    for name, value in actual.items():
        expected = torch.stack([state[name] for state in snapshots]).mean(dim=0)
        torch.testing.assert_close(value, expected)


@pytest.mark.unit
def test_post_step_hook_counts_only_backbone_optimizer_updates():
    torch.manual_seed(41)
    model = _model()
    average = model.averaged_backbone
    initial = deepcopy(model.backbone.state_dict())
    snapshots = [initial]
    backbone_optimizer = torch.optim.SGD(model.backbone.parameters(), lr=0.01)
    unrelated_parameter = torch.nn.Parameter(torch.tensor(1.0))
    unrelated_optimizer = torch.optim.SGD([unrelated_parameter], lr=0.1)
    callback = TS2VecAveraging()
    callback.on_fit_start(
        SimpleNamespace(optimizers=[backbone_optimizer, unrelated_optimizer]), model
    )

    assert average.n_averaged.item() == 1
    for step in range(3):
        _set_gradients(model.backbone, (step + 1) / 100)
        backbone_optimizer.step()
        backbone_optimizer.zero_grad(set_to_none=True)
        snapshots.append(deepcopy(model.backbone.state_dict()))
        assert average.n_averaged.item() == step + 2
        _assert_state_is_average(average.module.state_dict(), snapshots)

    unrelated_parameter.grad = torch.ones_like(unrelated_parameter)
    unrelated_optimizer.step()
    assert average.n_averaged.item() == 4

    callback.on_fit_end(SimpleNamespace(), model)
    _set_gradients(model.backbone, 0.2)
    backbone_optimizer.step()
    assert average.n_averaged.item() == 4


@pytest.mark.unit
def test_average_checkpoint_roundtrip_resumes_with_exact_running_average():
    torch.manual_seed(17)
    model = _model()
    optimizer = torch.optim.SGD(model.backbone.parameters(), lr=0.02)
    callback = TS2VecAveraging()
    callback.on_fit_start(SimpleNamespace(optimizers=[optimizer]), model)
    snapshots = [deepcopy(model.backbone.state_dict())]

    for value in (0.05, -0.02, 0.09):
        _set_gradients(model.backbone, value)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        snapshots.append(deepcopy(model.backbone.state_dict()))

    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)

    resumed = _model()
    resumed.load_state_dict(state)
    resumed_optimizer = torch.optim.SGD(resumed.backbone.parameters(), lr=0.02)
    resumed_callback = TS2VecAveraging()
    resumed_callback.on_fit_start(
        SimpleNamespace(optimizers=[resumed_optimizer]), resumed
    )
    assert resumed.averaged_backbone.n_averaged.item() == len(snapshots)
    _assert_state_is_average(resumed.averaged_backbone.module.state_dict(), snapshots)

    _set_gradients(resumed.backbone, -0.04)
    resumed_optimizer.step()
    snapshots.append(deepcopy(resumed.backbone.state_dict()))
    assert resumed.averaged_backbone.n_averaged.item() == len(snapshots)
    _assert_state_is_average(resumed.averaged_backbone.module.state_dict(), snapshots)
    resumed_callback.on_fit_end(SimpleNamespace(), resumed)
    callback.on_fit_end(SimpleNamespace(), model)


class _SeriesDataset(Dataset):
    """Tiny dict dataset for one short Trainer fit."""

    def __init__(self) -> None:
        self.series = torch.randn(6, 8, 2)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"series": self.series[index]}


class _CaptureBackbone(pl.Callback):
    """Capture the backbone after each actual training update."""

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[dict[str, torch.Tensor]] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.snapshots.append(deepcopy(pl_module.backbone.state_dict()))


@pytest.mark.unit
def test_tiny_trainer_average_includes_initial_state_and_each_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(53)
    config = get_config()
    monkeypatch.setattr(
        config,
        "_default_callbacks",
        {key: False for key in _DEFAULT_CALLBACK_REGISTRY},
    )
    monkeypatch.setattr(config, "_requeue_checkpoint", False)
    model = _model()
    snapshots = [deepcopy(model.backbone.state_dict())]
    capture = _CaptureBackbone()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = pl.Trainer(
            max_steps=3,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
            callbacks=[capture],
        )
        trainer.fit(model, DataLoader(_SeriesDataset(), batch_size=2))
    finally:
        torch.set_num_threads(previous_threads)

    snapshots.extend(capture.snapshots)
    assert trainer.global_step == 3
    assert len(capture.snapshots) == 3
    assert model.averaged_backbone.n_averaged.item() == 1 + trainer.global_step
    _assert_state_is_average(model.averaged_backbone.module.state_dict(), snapshots)


@pytest.mark.unit
def test_loss_keeps_float32_dot_products_inside_cpu_autocast():
    torch.manual_seed(67)
    z1 = torch.randn(2, 5, 4)
    z2 = torch.randn(2, 5, 4)
    loss = TS2VecLoss()
    expected = loss(z1, z2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss(z1, z2)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
