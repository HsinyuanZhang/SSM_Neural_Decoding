"""Custom Mamba3 sparse-state tuning; this is not original SDLoRA."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils import parametrize

from .peft import LoRALinear, _replace

_BIAS_NAMES = ("B_bias", "C_bias")


class DenseDelta(nn.Module):
    """Add a trainable dense offset while the base stays frozen."""

    def __init__(self, base: torch.Tensor) -> None:
        super().__init__()
        self.delta = nn.Parameter(base.new_zeros(base.shape))

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        return base + self.delta


class SparseDelta(nn.Module):
    """Add compact values only at fixed state indices."""

    def __init__(self, base: torch.Tensor, states: Sequence[int]) -> None:
        super().__init__()
        selected = [int(value) for value in states]
        if not selected:
            raise ValueError("at least one state must be selected")
        if len(selected) != len(set(selected)):
            raise ValueError("selected states must be unique")
        if min(selected) < 0 or max(selected) >= base.shape[-1]:
            raise ValueError("selected state is outside the bias state dimension")
        self.register_buffer("states", torch.tensor(selected, device=base.device, dtype=torch.long))
        self.values = nn.Parameter(base.new_zeros(*base.shape[:-1], len(selected)))

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        offset = base.new_zeros(base.shape)
        offset.index_copy_(-1, self.states, self.values)
        return base + offset


def _cores(model: nn.Module) -> list[tuple[int, nn.Module]]:
    try:
        return [(index, block.ssm) for index, block in enumerate(model.blocks)]
    except AttributeError as exc:
        raise ValueError("model must expose blocks with an ssm module") from exc


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _bias(core: nn.Module, name: str) -> torch.Tensor:
    if not hasattr(core, name):
        raise ValueError(f"Mamba3 core lacks required {name}")
    value = getattr(core, name)
    if value.ndim < 1:
        raise ValueError(f"{name} must have a state dimension")
    return value


def _receipt(value: dict[str, Any]) -> dict[str, Any]:
    json.dumps(value)
    return value


def _trainable_summary(model: nn.Module) -> tuple[list[str], int]:
    paths = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return paths, sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_unique_trainable_storage(model: nn.Module) -> None:
    """Reject duplicate trainable objects or overlapping trainable storage."""
    objects: dict[int, str] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        object_id = id(parameter)
        if object_id in objects:
            raise AssertionError(f"trainable parameter alias: {objects[object_id]} and {name}")
        objects[object_id] = name
        if parameter.numel() == 0:
            continue
        start = parameter.untyped_storage().data_ptr() + parameter.storage_offset() * parameter.element_size()
        stop = start + parameter.numel() * parameter.element_size()
        for old_start, old_stop, old_name in ranges:
            if start < old_stop and old_start < stop:
                raise AssertionError(f"trainable storage overlaps: {old_name} and {name}")
        ranges.append((start, stop, name))


def assert_stage_ready(model: nn.Module, stage: str) -> None:
    """Check that only the parameters for one sparse-tuning stage can train."""
    if stage not in {"dense_warmup", "selected", "sparse"}:
        raise ValueError("unknown sparse-state stage")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if stage == "selected":
        if trainable:
            raise AssertionError("selected stage must have no trainable parameters")
        return
    if stage == "dense_warmup":
        expected = {
            f"blocks.{index}.ssm.parametrizations.{name}.0.delta"
            for index, _ in _cores(model)
            for name in _BIAS_NAMES
        }
        if trainable != expected:
            raise AssertionError("dense warmup must train only B/C DenseDelta tensors")
        return
    if not trainable:
        raise AssertionError("sparse stage has no trainable compact adapters")
    allowed = (".parametrizations.B_bias.0.values", ".parametrizations.C_bias.0.values", ".A", ".B")
    if any(not name.endswith(allowed) for name in trainable):
        raise AssertionError("sparse stage has an unexpected trainable parameter")
    assert_unique_trainable_storage(model)


def begin_dense_selection(model: nn.Module) -> dict[str, Any]:
    """Freeze a decoder and install zero dense B/C deltas for warm-up."""
    _freeze_all(model)
    layers: list[int] = []
    for index, core in _cores(model):
        for name in _BIAS_NAMES:
            base = _bias(core, name)
            if parametrize.is_parametrized(core, name):
                raise ValueError(f"{name} already has a parametrization")
            parametrize.register_parametrization(core, name, DenseDelta(base))
        layers.append(index)
    paths, count = _trainable_summary(model)
    assert_unique_trainable_storage(model)
    model._sparse_state_warmup = {"stage": "dense_warmup", "dense_trainable_count": count}
    return _receipt({"stage": "dense_warmup", "layers": layers, "trainable_paths": paths,
                     "trainable_count": count, "custom_not_original_sdlora": True})


def selection_from_warmup(model: nn.Module, keep_states: int = 8) -> dict[str, Any]:
    """Select high-energy B/C dimensions and exactly restore frozen base biases."""
    if not isinstance(keep_states, int) or keep_states <= 0:
        raise ValueError("keep_states must be a positive integer")
    selections: dict[int, list[int]] = {}
    details: dict[int, dict[str, Any]] = {}
    for index, core in _cores(model):
        deltas: list[torch.Tensor] = []
        for name in _BIAS_NAMES:
            if not parametrize.is_parametrized(core, name):
                raise ValueError("dense warmup is not installed for every B/C bias")
            adapter = core.parametrizations[name][0]
            if not isinstance(adapter, DenseDelta):
                raise ValueError("B/C bias does not use DenseDelta")
            deltas.append(adapter.delta.detach())
        state_size = deltas[0].shape[-1]
        if deltas[1].shape[-1] != state_size:
            raise ValueError("B_bias and C_bias have different state sizes")
        if keep_states > state_size:
            raise ValueError("keep_states exceeds the state size")
        scores = sum(delta.square().sum(tuple(range(delta.ndim - 1))) for delta in deltas)
        selected = torch.topk(scores, keep_states, largest=True, sorted=True).indices.sort().values
        selected_list = selected.cpu().tolist()
        mask = torch.zeros(state_size, dtype=torch.bool, device=scores.device)
        mask.index_fill_(0, selected, True)
        selections[index] = selected_list
        details[index] = {"scores": scores.cpu().tolist(), "selected": selected_list,
                          "mask": mask.cpu().tolist(), "dense_delta_B": deltas[0].cpu().tolist(),
                          "dense_delta_C": deltas[1].cpu().tolist()}
        for name in _BIAS_NAMES:
            parametrize.remove_parametrizations(core, name, leave_parametrized=False)
    previous = getattr(model, "_sparse_state_warmup", {})
    model._sparse_state_warmup = {"stage": "selected",
                                  "dense_trainable_count": int(previous.get("dense_trainable_count", 0)),
                                  "selections": selections, "details": details}
    return _receipt({"stage": "selected", "algorithm": "sum B/C dense-delta squared over all non-state dimensions",
                     "selections": selections, "details": details, "custom_not_original_sdlora": True})


def _selection_for_layer(selections: Mapping[int | str, Sequence[int]], index: int) -> Sequence[int]:
    if index in selections:
        return selections[index]
    if str(index) in selections:
        return selections[str(index)]
    raise ValueError(f"missing selected states for layer {index}")


def install_sparse_state_tuning(model: nn.Module, selections: Mapping[int | str, Sequence[int]],
                                rank: int = 4) -> dict[str, Any]:
    """Install compact B/C offsets and root/core output LoRA adapters."""
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    _freeze_all(model)
    lora_paths: list[str] = []
    normalized: dict[int, list[int]] = {}
    for index, core in _cores(model):
        states = list(_selection_for_layer(selections, index))
        for name in _BIAS_NAMES:
            base = _bias(core, name)
            if parametrize.is_parametrized(core, name):
                raise ValueError(f"remove dense warmup before installing sparse {name}")
            parametrize.register_parametrization(core, name, SparseDelta(base, states))
        normalized[index] = [int(value) for value in states]
        path = f"blocks.{index}.ssm.out_proj"
        _replace(model, path, LoRALinear(core.out_proj, rank))
        lora_paths.append(path)
    for path in ("in_proj", "out_proj"):
        modules = dict(model.named_modules())
        if path not in modules or not isinstance(modules[path], nn.Linear):
            raise ValueError(f"model lacks root linear {path}")
        _replace(model, path, LoRALinear(modules[path], rank))
        lora_paths.append(path)
    paths, count = _trainable_summary(model)
    assert_unique_trainable_storage(model)
    warmup = getattr(model, "_sparse_state_warmup", {})
    return _receipt({"stage": "sparse", "rank": rank, "selections": normalized,
                     "lora_paths": lora_paths, "trainable_paths": paths, "trainable_count": count,
                     "warmup_dense_trainable_count": int(warmup.get("dense_trainable_count", 0)),
                     "warmup_dense_storage_count": int(warmup.get("dense_trainable_count", 0)),
                     "sparse_trainable_count": count, "sparse_storage_count": count,
                     "custom_not_original_sdlora": True,
                     "description": "custom sparse-state port; not original SDLoRA"})
