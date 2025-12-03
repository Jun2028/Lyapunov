"""
Lightweight evaluator for PPO fine-tuning.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Dict, Optional

import torch

from logging import getLogger

from src.utils import to_cuda, MyTimeoutError
from .reward import LyapunovReward
from .utils import slice_batch_sequences


logger = getLogger(__name__)


class RLEvaluator:
    """
    Evaluate the current policy by greedily decoding and checking Lyapunov validity.
    """

    def __init__(self, modules, env, params, data_path) -> None:
        if data_path is None:
            raise ValueError("Evaluation requires --reload_data so that reference targets are available.")
        self.modules = modules
        self.env = env
        self.params = params
        self.data_path = data_path
        self.reward_helper = LyapunovReward(env)

    def run(self, data_type: str = "valid", tasks: Optional[list[str]] = None, max_batches: Optional[int] = None) -> Dict[str, float]:
        encoder = self.modules["encoder"].module if self.params.multi_gpu else self.modules["encoder"]
        decoder = self.modules["decoder"].module if self.params.multi_gpu else self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        results = OrderedDict()

        eval_tasks = tasks or self.params.tasks
        with torch.no_grad():
            for task in eval_tasks:
                if data_type == "valid":
                    indices = [1]
                else:
                    indices = list(range(2, len(self.data_path[task])))
                if not indices:
                    logger.warning(f"No data split '{data_type}' available for task {task}.")
                    continue
                total = 0
                valid = 0
                codes = Counter()

                for data_path_idx in indices:
                    iterator = self.env.create_test_iterator(
                        data_type,
                        task,
                        data_path=self.data_path,
                        data_path_idx=data_path_idx,
                        batch_size=self.params.batch_size_eval,
                        params=self.params,
                        size=None,
                    )

                    for batch_idx, ((x1, len1), (x2, len2), _) in enumerate(iterator):
                        if max_batches is not None and batch_idx >= max_batches:
                            break

                        src_refs = slice_batch_sequences(x1, len1)
                        tgt_refs = slice_batch_sequences(x2, len2)
                        self.reward_helper.set_reference_batch(src_refs, tgt_refs)
                        src_words = self.reward_helper._src_cache or []
                        tgt_words = self.reward_helper._tgt_cache or []

                        x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)
                        encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                        src_enc = encoded.transpose(0, 1)
                        generated, gen_len = decoder.generate(src_enc, len1, max_len=self.params.max_len)
                        generated_cpu = generated.cpu()
                        gen_len_cpu = gen_len.cpu()

                        for idx in range(len1.size(0)):
                            src = src_words[idx]
                            tgt = tgt_words[idx]
                            hyp_ids = generated_cpu[: gen_len_cpu[idx], idx]
                            hyp = self.reward_helper._prepare_reference(hyp_ids)
                            if not hyp:
                                code = -2
                            else:
                                try:
                                    code = self.env.check_lyap_validity(src, hyp, tgt)
                                except MyTimeoutError:
                                    code = -3
                                except Exception:
                                    code = -4
                            codes[code] += 1
                            total += 1
                            if code == 1:
                                valid += 1

                accuracy = valid / total if total > 0 else 0.0
                prefix = f"{data_type}_{task}"
                results[f"{prefix}_accuracy"] = accuracy
                for code, count in sorted(codes.items()):
                    results[f"{prefix}_count_{code}"] = count

        return results
