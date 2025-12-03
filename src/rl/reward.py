"""
Reward helpers for PPO fine-tuning.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch

from ..utils import MyTimeoutError


TokenLike = Union[Sequence[int], Sequence[str], torch.Tensor] #should be one of: list of ints, list of strings, torch tensor


class RewardFunction:
    """
    Base interface for evaluating sampled sequences.
    """

    def __call__(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LyapunovReward(RewardFunction):
    """
    Wraps the Lyapunov validator so PPO can compute terminal rewards quickly.
    """

    def __init__(self, env, reward_scale: float = 1.0) -> None:
        """
        Args:
            env: Lyapunov environment that exposes token dictionaries and validators.
            reward_scale: Scalar to multiply the raw validator output by.
        """
        self.env = env
        self.reward_scale = reward_scale
        self._src_cache: List[List[str]] | None = None
        self._tgt_cache: List[List[str]] | None = None

    def set_reference_batch(self, src_batch: Sequence[TokenLike], tgt_batch: Sequence[TokenLike]) -> None:
        """Cache raw (src, tgt) sequences for the next reward computation step."""
        if len(src_batch) != len(tgt_batch):
            raise ValueError("src_batch and tgt_batch must have identical batch sizes.")
        self._src_cache = [self._prepare_reference(seq) for seq in src_batch]
        self._tgt_cache = [self._prepare_reference(seq) for seq in tgt_batch]

    def __call__(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: (batch, seq_len) token ids.
            lengths: (batch,) number of valid tokens per sequence.
        Returns:
            Reward tensor (batch,) scaled to reward_scale.
        """

        if self._src_cache is None or self._tgt_cache is None:
            raise RuntimeError("Reference batch not set. Call set_reference_batch() before computing rewards.")
        batch = sequences.size(0)
        if len(self._src_cache) != batch or len(self._tgt_cache) != batch:
            raise ValueError("Cached (src, tgt) batches must match reward batch size.")

        rewards = torch.zeros(batch, dtype=torch.float32, device=sequences.device)
        for idx in range(batch):
            length = lengths[idx].item()
            tokens = sequences[idx, :length].tolist()
            hyp = self._prepare_reference(tokens)
            if not hyp:
                continue
            src = self._src_cache[idx]
            tgt = self._tgt_cache[idx] #the ODE system src and the reference Lyapunov tgt, (src, tgt) comes from the (heldout) supervised dataset.
            try:
                result = self.env.check_lyap_validity(
                    src,
                    hyp,
                    tgt,
                )  # recall that check_lyap_validity can return (1 valid, 0 invalid, -1 optim error, -2 bad hyp, -3 timeout, -4 other exception, -5 input error)
            except MyTimeoutError:
                result = -3
            except Exception:
                result = -4
            rewards[idx] = self._map_result_to_reward(result)
        return rewards * self.reward_scale

    def _prepare_reference(self, seq: TokenLike) -> List[str]:
        """
        Convert an arbitrary token representation into a cleaned list of words.

        Args:
            seq: Sequence of ids/strings/torch tensor representing a sample.

        Returns:
            List of tokens with padding and boundary symbols removed.
        """
        words = self._to_words(seq)
        return self._strip_special(words)

    def _to_words(self, seq: TokenLike) -> List[str]:
        """
        Map tokens (indices or already-decoded words) into the environment vocabulary.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        words: List[str] = []
        for tok in seq:
            if isinstance(tok, str):
                word = tok
            else:
                word = self.env.id2word[int(tok)]
            words.append(word)
        return words

    def _strip_special(self, words: List[str]) -> List[str]:
        """
        Remove BOS / EOS / PAD markers, returning the symbolic content only. Preprocessing for token -> expression.
        """
        if not words:
            return []
        eos = self.env.id2word[self.env.eos_index]
        pad = self.env.id2word[self.env.pad_index]
        cleaned = [w for w in words if w != pad]
        if cleaned and cleaned[0] == eos:
            cleaned = cleaned[1:]
        while cleaned and cleaned[-1] == eos:
            cleaned.pop()
        return cleaned

    def _map_result_to_reward(self, result: float) -> float:
        """
        Convert validator exit codes into bounded rewards to stabilize PPO.
        """
        reward_table = {
            1: 1.0,   # Valid Lyapunov
            0: 0.0,   # Incorrect but well-formed
            -1: -0.3, # SOS / optimizer failure
            -2: -0.5, # Ill-formed hypothesis
            -3: -0.1, # Timeout
            -4: -0.4, # Unknown checker error
            -5: -0.6, # Ill-formed input system
        }
        return reward_table.get(int(result), -0.2) #map any other unknown result to -0.2
