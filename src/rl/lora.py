"""
Lightweight LoRA adapters for the transformer decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    attention_linear_names: Tuple[str, ...] = ("q_lin", "k_lin", "v_lin", "out_lin")
    ffn_linear_names: Tuple[str, ...] = ("lin1", "lin2")


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear layer with a frozen base weight plus trainable low-rank adapters.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self._mark_trainable(self.lora_A.weight)
        self._mark_trainable(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.base(x)
        update = self.lora_B(self.lora_A(self.dropout(x)))
        return residual + self.scaling * update

    def lora_parameters(self) -> Iterable[torch.nn.Parameter]:
        yield from self.lora_A.parameters()
        yield from self.lora_B.parameters()

    @staticmethod
    def _mark_trainable(param: torch.nn.Parameter) -> None:
        param.requires_grad = True
        setattr(param, "_is_lora_param", True)


def apply_lora_to_decoder(decoder: nn.Module, config: LoRAConfig) -> List[LoRALinear]:
    """
    Inject LoRA adapters into the decoder in-place.
    Returns the list of injected LoRA modules for easy parameter management.
    """

    module = getattr(decoder, "module", decoder) #decoder is wrapped by nn.parallel.DistributedDataParallel, we are peeling it off here
    adapters: List[LoRALinear] = []
    for attention in getattr(module, "attentions", []):
        adapters.extend(_maybe_wrap_linear(attention, config.attention_linear_names, config))
    for ffn in getattr(module, "ffns", []):
        adapters.extend(_maybe_wrap_linear(ffn, config.ffn_linear_names, config))
    return [adapter for adapter in adapters if adapter is not None]


def only_lora_parameters(module: nn.Module) -> List[torch.nn.Parameter]:
    """
    Return all parameters tagged as LoRA trainables (used to build a dedicated optimizer). fine-tune with LoRA by passing only_lora_parameters(model) to your optimizer instead of model.parameters().
    """

    params: List[torch.nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, LoRALinear):
            params.extend(list(m.lora_parameters()))
    return params


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only the LoRA adapter tensors from the provided module.
    """

    module = getattr(module, "module", module)
    state: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        if getattr(param, "_is_lora_param", False):
            state[name] = param.detach().cpu()
    return state


def load_lora_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Load previously saved LoRA adapter tensors into the module.
    """

    module = getattr(module, "module", module)
    named_params = dict(module.named_parameters())
    for name, tensor in state_dict.items():
        if name not in named_params:
            raise KeyError(f"LoRA parameter {name} not found in target module.")
        named_params[name].data.copy_(tensor.to(named_params[name].device))


def _maybe_wrap_linear(parent: nn.Module, attr_names: Sequence[str], config: LoRAConfig) -> List[LoRALinear]:
    adapters: List[LoRALinear] = []
    for name in attr_names:
        layer = getattr(parent, name, None)
        if layer is None or isinstance(layer, LoRALinear):
            continue
        if not isinstance(layer, nn.Linear):
            continue
        adapter = LoRALinear(layer, config.rank, config.alpha, config.dropout)
        try:
            device = next(layer.parameters()).device
            adapter.to(device)
        except StopIteration:
            pass
        setattr(parent, name, adapter)
        adapters.append(adapter)
    return adapters
