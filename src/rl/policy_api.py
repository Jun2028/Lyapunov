"""
Utilities to adapt the transformer decoder for PPO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from logging import getLogger


logger = getLogger(__name__)


@dataclass
class PolicyOutput:
    logits: torch.Tensor #before softmax
    values: torch.Tensor
    hidden_states: torch.Tensor


class TransformerPolicy(nn.Module):
    """
    Thin wrapper that exposes policy / value heads on top of the decoder.
    """

    def __init__(
        self,
        decoder: nn.Module,
        value_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.decoder_module = getattr(decoder, "module", decoder)
        if value_head is None:
            value_head = nn.Linear(self.decoder_module.dim, 1)
        self.value_head = value_head
        self.pad_index = self.decoder_module.pad_index

    @torch.no_grad()
    def clone_policy(self) -> "TransformerPolicy":
        """
        Snapshot the current policy for KL comparisons.
        """

        reference_decoder = copy.deepcopy(self.decoder_module)
        reference_value = copy.deepcopy(self.value_head)
        reference = TransformerPolicy(reference_decoder, reference_value)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        return reference

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        src_enc: Optional[torch.Tensor] = None,
        src_len: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> PolicyOutput:
        """
        Args:
            input_ids: (seq_len, batch) tokens including BOS.
            lengths: (batch,) valid lengths.
            src_enc / src_len: optional encoder context for conditional generation.
        """

        hidden = self.decoder(
            "fwd",
            x=input_ids,
            lengths=lengths,
            causal=causal,
            src_enc=src_enc,
            src_len=src_len,
            positions=None,
        )
        hidden = hidden.transpose(0, 1)  # (batch, seq_len, dim)

        flat_hidden = hidden.reshape(-1, hidden.size(-1)) # recall that -1 infers the first dimension (batch * seq_len)
        logits = self.decoder_module.proj(flat_hidden).view(hidden.size(0), hidden.size(1), -1) # (batch, seq_len, vocab_size)
        value_hidden = flat_hidden
        if value_hidden.dtype != self.value_head.weight.dtype:
            value_hidden = value_hidden.to(self.value_head.weight.dtype)
        values = self.value_head(value_hidden).view(hidden.size(0), hidden.size(1)).to(hidden.dtype)

        return PolicyOutput(logits=logits, values=values, hidden_states=hidden)

    def compute_log_probs(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            actions: (batch, seq_len) token ids actually sampled.
        """

        log_probs = F.log_softmax(logits.float(), dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) # (batch, seq_len) action_log_probs[i, t] is the log-prob assigned by the model 
        return action_log_probs.to(logits.dtype)                                                            # to the actual action taken at timestep t in batch i.

    def entropy(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return the mean token-level entropy of the policy distribution.
        Args:
            logits: Raw decoder scores, shape (batch, seq_len, vocab_size).
            mask: Optional binary mask (batch, seq_len) to ignore padding tokens.
        """
        ent = -(F.log_softmax(logits.float(), dim=-1) * F.softmax(logits.float(), dim=-1)).sum(dim=-1)
        if mask is not None:
            mask = mask.to(ent.dtype)
            total = mask.sum().clamp_min(1)
            return (ent * mask).sum() / total
        return ent.mean()

    def kl_divergence(
        self,
        new_logits: torch.Tensor,
        old_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL(new || old) per token.
        """

        new_log_probs = F.log_softmax(new_logits.float(), dim=-1)
        old_log_probs = F.log_softmax(old_logits.float(), dim=-1)
        kl = torch.exp(new_log_probs) * (new_log_probs - old_log_probs)
        kl = kl.sum(dim=-1)
        if mask is not None:
            kl = kl * mask
        return kl
