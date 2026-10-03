"""CPU checks for the input-adaptation interface and custom state-offset ports."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ssm_decode.input_adaptation as adapt
from ssm_decode.peft import LoRALinear


def _kernel(**items):
    q, k, v, z = items["Q"], items["K"], items["V"], items["Z"]
    state = torch.zeros_like(v[:, 0])
    result = []
    for index in range(v.shape[1]):
        state = .5 * state + v[:, index] + q[:, index].mean(-1, keepdim=True) + k[:, index].mean(-1, keepdim=True)
        result.append(torch.nn.functional.silu(z[:, index]) * state)
    return torch.stack(result, 1)


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_inner, self.d_state = 4, 2
        self.num_bc_heads, self.mimo_rank = 1, 1
        self.nheads, self.headdim, self.num_rope_angles = 1, 4, 1
        self.is_mimo, self.is_outproj_norm = False, False
        self.A_floor, self.chunk_size = .01, 8
        self.in_proj = nn.Linear(4, 16, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.B_norm = nn.Identity()
        self.C_norm = nn.Identity()
        self.B_bias = nn.Parameter(torch.zeros(1, 1, 2))
        self.C_bias = nn.Parameter(torch.zeros(1, 1, 2))
        self.dt_bias = nn.Parameter(torch.zeros(1))
        self.D = nn.Parameter(torch.ones(1, 4))

    def forward(self, u):
        batch, length, _ = u.shape
        z, x, b, c, dt, a, trap, angles = torch.split(self.in_proj(u), [4, 4, 2, 2, 1, 1, 1, 1], -1)
        z, x = z.view(batch, length, 1, 4), x.view(batch, length, 1, 4)
        a = -(a.clamp_min(0) + torch.reciprocal(1 - a.clamp_max(0))).clamp_max(-self.A_floor)
        y = _kernel(Q=c.view(batch, length, 1, 2), K=b.view(batch, length, 1, 2), V=x,
                    ADT=(a * torch.nn.functional.softplus(dt + self.dt_bias)).transpose(1, 2),
                    DT=dt.transpose(1, 2), Trap=trap.transpose(1, 2), Q_bias=self.C_bias.squeeze(1),
                    K_bias=self.B_bias.squeeze(1), Angles=angles.unsqueeze(-2), D=self.D, Z=z,
                    chunk_size=8, Input_States=None, return_final_states=False, cu_seqlens=None)
        return self.out_proj(y.flatten(-2))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssm = Core()
        self.norm1, self.norm2 = nn.LayerNorm(4), nn.LayerNorm(4)
        self.ffn, self.dropout = nn.Linear(4, 4), nn.Identity()

    def forward(self, x):
        x = x + self.dropout(self.ssm(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class Model(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.config = SimpleNamespace(input_size=3)
        self.in_proj, self.out_proj = nn.Linear(3, 4), nn.Linear(4, 2)
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.final_norm = nn.LayerNorm(4)

    def forward(self, x):
        hidden = self.in_proj(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out_proj(self.final_norm(hidden))


@pytest.fixture(autouse=True)
def kernel(monkeypatch):
    monkeypatch.setattr(adapt, "_siso_kernel", _kernel)


@pytest.mark.parametrize("method", ["none", "io", "full", "affine"])
def test_startup_is_exact_and_basic_freeze_policy(method):
    torch.manual_seed(8)
    model, x = Model(), torch.randn(2, 5, 3)
    baseline = model(x).detach()
    receipt = adapt.configure_adaptation(model, method)
    assert torch.equal(model(x), baseline)
    assert receipt["trainable_count"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    if method == "none":
        assert not receipt["trainable_paths"]
    elif method == "affine":
        assert receipt["trainable_paths"] == ["input_adaptation.gain", "input_adaptation.bias"]
    else:
        assert all(not name.startswith("blocks.") for name in receipt["trainable_paths"])


@pytest.mark.parametrize("scope, expected", [("all", {"in_proj", "out_proj", "blocks"}), ("root", {"in_proj", "out_proj"}), ("core", {"blocks"}), ("input", {"in_proj"})])
def test_lora_scopes_and_input_bias(scope, expected):
    model = Model()
    receipt = adapt.configure_adaptation(model, "lora", rank=4, alpha=4, lora_scope=scope, train_input_bias=True)
    names = set(receipt["trainable_paths"])
    assert all(any(name.startswith(prefix) for prefix in expected) for name in names if name.endswith((".A", ".B")))
    assert isinstance(model.in_proj, LoRALinear) is (scope != "core")
    assert receipt["root_input_bias_trainable"] is (scope != "core")
    if scope != "core":
        assert "in_proj.base.bias" in names
    if scope == "input":
        assert not isinstance(model.out_proj, LoRALinear)
    assert json_safe(receipt)


def json_safe(value):
    import json
    json.dumps(value)
    return True


@pytest.mark.parametrize("rank", [1, 4, 16])
def test_affine_lora_replay_and_merged_export(rank):
    torch.manual_seed(rank)
    source, x = Model(), torch.randn(2, 5, 3)
    initial = copy.deepcopy(source.state_dict())
    adapt.configure_adaptation(source, "affine_lora", rank=rank, lora_scope="root")
    source.input_adaptation.gain.data.copy_(torch.tensor([1.2, .8, 1.1]))
    source.input_adaptation.bias.data.copy_(torch.tensor([.2, -.1, .3]))
    source.in_proj.A.data.normal_()
    source.out_proj.A.data.normal_()
    live = source(x).detach()
    replay = Model()
    replay.load_state_dict(initial)
    adapt.configure_adaptation(replay, "affine_lora", rank=rank, lora_scope="root")
    replay.load_state_dict(source.state_dict())
    assert torch.equal(replay(x), live)
    exported = adapt.merged_adapted_state_dict(source)
    assert all("input_adaptation" not in key and not key.endswith((".A", ".B", ".rows")) for key in exported)
    folded = Model()
    folded.load_state_dict(exported)
    assert torch.allclose(folded(x), live, atol=2e-6, rtol=2e-6)


def test_original_and_rotated_coordinates_have_known_angle_difference():
    c = torch.tensor([[[[1., 0., 7., 11.]]]])
    bias = torch.zeros(1, 4)
    angles = torch.tensor([[[[torch.atanh(torch.tensor(.5))]]]])
    dt = torch.ones(1, 1, 1)
    original = adapt.normalized_readout(c, bias)
    rotated = adapt.rotated_readout(c, bias, angles, dt)
    theta = (torch.tanh(angles.to(torch.bfloat16).float()) * torch.pi).to(torch.bfloat16).float()
    expected = torch.tensor([[[[torch.cos(theta[0, 0, 0, 0]), torch.sin(theta[0, 0, 0, 0]), 7., 11.]]]]).to(torch.bfloat16).float()
    assert torch.equal(rotated, expected)
    # At Phi=pi/2, q_rot=R(+Phi)q. Its dot product maps h to R(-Phi)h.
    h = torch.tensor([3., 5., 13., 17.])
    rotated_dot = (rotated[0, 0, 0] * h).sum()
    physical = torch.stack((h[1], -h[0], h[2], h[3]))
    original_dot = (original[0, 0, 0] * physical).sum()
    assert torch.allclose(rotated_dot, original_dot, atol=3e-2, rtol=0)
    zero = adapt.rotated_readout(c, bias, torch.zeros_like(angles), dt)
    assert torch.equal(zero, original)
    assert torch.equal(rotated[..., 2:], original[..., 2:])
    assert not torch.equal(original, rotated)


def test_zero_angle_fractional_original_and_rotated_coordinates_are_bit_equal():
    c = torch.tensor([[[[.101, -.203, .307, -.409]]]])
    bias = torch.tensor([[.017, -.029, .043, -.061]])
    angles = torch.zeros(1, 1, 1, 1)
    dt = torch.ones(1, 1, 1)
    original = adapt.original_readout(c, bias)
    rotated = adapt.rotated_readout(c, bias, angles, dt)
    assert torch.equal(original, rotated)
    assert not torch.equal(adapt.normalized_readout(c, bias), original)


@pytest.mark.parametrize("method, coordinate", [("offset_original", "original"), ("offset_rotated", "rotated")])
def test_offset_is_exact_at_start_and_rejects_stream_apis(method, coordinate):
    torch.manual_seed(4)
    model, x = Model(1), torch.randn(2, 5, 3)
    baseline = model(x).detach()
    receipt = adapt.configure_adaptation(model, method, rank=4)
    assert torch.equal(model(x), baseline)
    assert receipt["coordinate"] == coordinate
    assert all(name.startswith(("input_adaptation", "blocks.0.ssm.state_offset")) for name in receipt["trainable_paths"])
    with pytest.raises(ValueError):
        model.blocks[0].ssm(torch.randn(1, 2, 4), seq_idx=torch.zeros(1, 2, dtype=torch.long))
    with pytest.raises(ValueError):
        adapt.merged_adapted_state_dict(model)


def test_future_input_does_not_change_prior_outputs():
    model, x = Model(1), torch.randn(2, 6, 3)
    adapt.configure_adaptation(model, "offset_original", rank=1)
    model.blocks[0].ssm.state_offset.U.data.fill_(1)
    changed = x.clone()
    changed[:, 4:] += 20
    assert torch.equal(model(x)[:, :4], model(changed)[:, :4])


def test_offset_rejects_non_siso_core_and_invalid_rank():
    model = Model(1)
    model.blocks[0].ssm.is_mimo = True
    with pytest.raises(ValueError, match="SISO"):
        adapt.configure_adaptation(model, "offset_original", rank=1)
    with pytest.raises(ValueError, match="rank"):
        adapt.configure_adaptation(Model(1), "affine", rank=0)
