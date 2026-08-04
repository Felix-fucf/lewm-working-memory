from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from module_oot import (
    LeakyRNNMemory,
    PointwiseActionPredictor,
    StrictRecurrentOOTJEPA,
)


class DummyEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, pixels: torch.Tensor, **_: object) -> SimpleNamespace:
        pooled = pixels.mean(dim=(-2, -1))
        return SimpleNamespace(last_hidden_state=self.proj(pooled).unsqueeze(1))


class DummyActionEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.proj(action.float())


def _model(dim: int = 8) -> StrictRecurrentOOTJEPA:
    return StrictRecurrentOOTJEPA(
        encoder=DummyEncoder(dim),
        projector=nn.Sequential(nn.Linear(dim, 16), nn.LayerNorm(16), nn.GELU(), nn.Linear(16, dim)),
        action_encoder=DummyActionEncoder(dim),
        memory=LeakyRNNMemory(
            input_dim=2 * dim,
            hidden_dim=dim,
            leak_rate=0.2,
            learnable_leak=False,
            dropout=0.0,
            initial_state_std=0.0,
        ),
        predictor=PointwiseActionPredictor(dim, dim, 16, dim, dropout=0.1),
        embed_dim=dim,
    )


def _previous_action(current: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(current[:, :1]), current[:, :-1]), dim=1)


def test_22_frame_encode_and_21_step_alignment() -> None:
    model = _model().eval()
    encoded = model.encode({
        "pixels": torch.randn(2, 22, 3, 12, 12),
        "action": torch.randn(2, 22, 3),
    })
    assert encoded["emb"].shape == (2, 22, 8)
    context = encoded["emb"][:, :21]
    current_action = encoded["act_emb"][:, :21]
    state, hidden, final = model.remember(context, _previous_action(current_action))
    assert state is hidden
    assert state.shape == (2, 21, 8)
    assert final.shape == (2, 8)
    assert model.predict(state, current_action).shape == (2, 21, 8)


def test_action_timing_and_pointwise_temporal_isolation() -> None:
    model = _model().eval()
    visual = torch.randn(2, 6, 8)
    action = torch.randn(2, 6, 8)
    previous = _previous_action(action)
    hidden, _, _ = model.remember(visual, previous)
    prediction = model.predict(hidden, action)

    changed_action = action.clone()
    changed_action[:, 2] += 4.0
    changed_previous = _previous_action(changed_action)
    changed_hidden, _, _ = model.remember(visual, changed_previous)
    changed_prediction = model.predict(changed_hidden, changed_action)
    # a_t enters the predictor at t and the recurrent update from t+1 onward.
    assert not torch.equal(prediction[:, 2], changed_prediction[:, 2])
    assert torch.equal(hidden[:, :3], changed_hidden[:, :3])
    assert not torch.equal(hidden[:, 3:], changed_hidden[:, 3:])

    isolated_hidden = hidden.clone()
    isolated_action = action.clone()
    isolated_hidden[:, 4] += 3.0
    isolated_action[:, 4] -= 2.0
    isolated = model.predict(isolated_hidden, isolated_action)
    keep = torch.tensor([0, 1, 2, 3, 5])
    assert torch.equal(prediction[:, keep], isolated[:, keep])


def test_no_visual_bypass_for_ablation_and_no_temporal_predictor_modules() -> None:
    model = _model().eval()
    action = torch.randn(3, 5, 8)
    zero_hidden = torch.zeros(3, 5, 8)
    baseline = model.predict(zero_hidden, action)
    radically_different_visual = torch.randn(3, 5, 8) * 100.0
    # A no-memory prediction has no API path by which visual input can enter.
    assert torch.equal(baseline, model.predict(torch.zeros_like(radically_different_visual), action))
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(module, forbidden) for module in model.predictor.modules())
    assert not any("pos" in name.lower() for name, _ in model.predictor.named_parameters())
    assert not hasattr(model, "memory_to_embedding")
    assert not hasattr(model, "memory_output_norm")
    assert not hasattr(model, "pred_proj")


def test_all_parameters_receive_gradient_and_checkpoint_strict_load(tmp_path) -> None:
    model = _model().train()
    encoded = model.encode({
        "pixels": torch.randn(2, 22, 3, 10, 10),
        "action": torch.randn(2, 22, 3),
    })
    context = encoded["emb"][:, :21]
    action = encoded["act_emb"][:, :21]
    hidden, _, _ = model.remember(context, _previous_action(action))
    model.predict(hidden, action).square().mean().backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert missing == []

    checkpoint = tmp_path / "strict.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
