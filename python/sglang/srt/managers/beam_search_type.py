# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam search data structures for managing beam search state.

This module defines the core data structures used in beam search:
- BeamSearchSequence: Represents a single beam candidate sequence
- BeamSearchList: Manages the collection of beam candidates for a request
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchSequence:
    """A single beam candidate sequence in beam search.

    This class tracks tokens and log probabilities for one beam candidate.
    Dense token storage may live on BeamSearchList.token_ids during decode;
    the tokens list is materialized for stop checks, finish, and user-facing output.

    The text field is optional and only filled when the sequence is about to be
    returned to the user.
    """

    tokens: List[int]  # Generated tokens (excluding prompt)
    cum_logprob: float = 0.0  # Cumulative log probability for sorting

    finish_reason: Optional[object] = None  # Reason for completion (if finished)
    text: Optional[str] = None  # Decoded text (filled on completion)
    beam_score: Optional[float] = None  # Beam search score, for return
    trie_node: Optional[object] = None  # Current trie node for prefix-constrained beam search
    is_dummy: bool = False  # 是否为用于填充的 dummy beam

    def finished(self):
        return self.finish_reason is not None


@dataclass
class BeamSearchList:
    """Manages the collection of beam candidates for a beam search request.

    This class maintains both object-based (BeamSearchSequence) and tensor-based
    representations for efficient computation. Tensor fields enable parallel
    operations while BeamSearchSequence objects store complex state.

    Attributes:
        batch_slot_start_idx: Starting index in batch.req_pool_indices array where
            this beam group's incomplete beams are stored (consecutive beam_width slots)
        completed: List of finished beam sequences
        incomplete: List of active beam sequences still being explored

        # Tensor-based state for parallel operations on incomplete beams:
        cum_logprobs: Cumulative log probabilities, Shape: [num_incomplete_beams]
        last_tokens: Last token of each beam (updated when incomplete refreshes),
            Shape: [num_incomplete_beams]
        prompt_lens: Prompt lengths for KV cache (set only at beamlist construction),
            Shape: [num_incomplete_beams]
        token_ids: Dense generated tokens for incomplete beams,
            Shape: [beam_width, max_new_tokens]
        cur_len: Number of valid columns in token_ids
        dense_authoritative: When True, ``generated_len`` trusts ``cur_len`` and
            incomplete may hold empty token stubs (fast path).
    """

    batch_slot_start_idx: int = -1
    completed: List[BeamSearchSequence] = field(default_factory=list)
    incomplete: List[BeamSearchSequence] = field(default_factory=list)
    has_dummy: bool = False  # incomplete 列表中是否包含 dummy 填充 beam

    # Tensor-based state for parallel operations (only for incomplete beams)
    cum_logprobs: Optional[torch.Tensor] = None
    last_tokens: Optional[torch.Tensor] = None
    prompt_lens: Optional[torch.Tensor] = None
    token_ids: Optional[torch.Tensor] = None
    cur_len: int = 0
    dense_authoritative: bool = False

    def empty(self):
        return len(self.completed) + len(self.incomplete) == 0

    def generated_len(self) -> int:
        """Current generated token length for incomplete beams.

        Fast dense path (``dense_authoritative``): trust ``cur_len``.
        Slow paths that clear dense state update ``incomplete[].tokens``;
        prefer list length there.
        """
        if self.dense_authoritative and self.token_ids is not None and self.cur_len > 0:
            return int(self.cur_len)
        if self.incomplete:
            return len(self.incomplete[0].tokens)
        if self.token_ids is not None and self.cur_len > 0:
            return int(self.cur_len)
        if self.completed:
            return len(self.completed[0].tokens)
        return 0

    def init_token_ids(
        self, max_new_tokens: int, device: Optional[torch.device] = None
    ) -> None:
        """Allocate a dense token buffer and seed it from incomplete sequences."""
        beam_width = len(self.incomplete)
        max_new_tokens = max(int(max_new_tokens), 1)
        if device is None:
            if self.last_tokens is not None:
                device = self.last_tokens.device
            else:
                device = torch.device("cpu")
        self.token_ids = torch.zeros(
            (beam_width, max_new_tokens), dtype=torch.int64, device=device
        )
        self.cur_len = 0
        self.dense_authoritative = True
        if beam_width == 0:
            return

        for i, beam in enumerate(self.incomplete):
            n = len(beam.tokens)
            if n <= 0:
                continue
            if n > max_new_tokens:
                n = max_new_tokens
            self.token_ids[i, :n] = torch.as_tensor(
                beam.tokens[:n], dtype=torch.int64, device=device
            )
            self.cur_len = max(self.cur_len, n)

    def expand_token_ids(
        self,
        parent_indices: Union[Sequence[int], torch.Tensor],
        new_tokens: Union[Sequence[int], torch.Tensor],
    ) -> None:
        """Gather parent rows and append new tokens.

        Parents are clamped to ``[0, beam_width)`` before gather to avoid
        IndexKernel OOB. Works on CPU or CUDA buffers.
        """
        if self.token_ids is None:
            return

        if isinstance(parent_indices, torch.Tensor):
            parents = parent_indices.to(
                device=self.token_ids.device, dtype=torch.int64
            ).view(-1)
        else:
            parents = torch.as_tensor(
                parent_indices, dtype=torch.int64, device=self.token_ids.device
            )

        n = int(parents.numel())
        if n == 0:
            self.token_ids = self.token_ids[:0]
            self.dense_authoritative = True
            return

        if isinstance(new_tokens, torch.Tensor):
            toks = new_tokens.to(
                device=self.token_ids.device, dtype=self.token_ids.dtype
            ).view(-1)
        else:
            toks = torch.as_tensor(
                new_tokens, dtype=self.token_ids.dtype, device=self.token_ids.device
            )

        if toks.numel() != n:
            raise ValueError("parent_indices and new_tokens must have the same length")

        max_parent = max(self.token_ids.shape[0] - 1, 0)
        parents = parents.clamp(min=0, max=max_parent)
        gathered = self.token_ids[parents]
        col = int(self.cur_len)
        if col >= gathered.shape[1]:
            gathered = torch.nn.functional.pad(gathered, (0, 1))
        gathered[:, col] = toks
        self.token_ids = gathered
        self.cur_len = col + 1
        self.dense_authoritative = True

    def sequences_from_token_ids(
        self,
        cum_logprobs: Optional[Union[Sequence[float], torch.Tensor]] = None,
    ) -> List[BeamSearchSequence]:
        """Build BeamSearchSequence objects from the dense token buffer."""
        if self.token_ids is None or self.cur_len <= 0:
            return []

        tokens_cpu = self.token_ids[:, : self.cur_len].detach().cpu().tolist()
        if cum_logprobs is None:
            if self.cum_logprobs is not None:
                vals = self.cum_logprobs.detach().cpu().tolist()
            else:
                vals = [0.0] * len(tokens_cpu)
        elif isinstance(cum_logprobs, torch.Tensor):
            vals = cum_logprobs.detach().cpu().tolist()
        else:
            vals = list(cum_logprobs)

        return [
            BeamSearchSequence(tokens=toks, cum_logprob=float(val))
            for toks, val in zip(tokens_cpu, vals)
        ]

    def ensure_empty_stubs(self, n: int) -> List[BeamSearchSequence]:
        """Ensure ``incomplete`` holds ``n`` empty-token stubs without D2H."""
        if (
            self.incomplete
            and len(self.incomplete) == n
            and all(not beam.tokens for beam in self.incomplete)
        ):
            return self.incomplete
        return [BeamSearchSequence(tokens=[], cum_logprob=0.0) for _ in range(n)]

    def clear_dense(self) -> None:
        self.token_ids = None
        self.cur_len = 0
        self.dense_authoritative = False

    def materialize_incomplete_if_needed(self) -> None:
        """If incomplete holds empty stubs, rebuild tokens from dense buffer."""
        if not (
            self.dense_authoritative
            and isinstance(self.token_ids, torch.Tensor)
            and self.cur_len > 0
            and self.incomplete
            and all(not beam.tokens for beam in self.incomplete)
        ):
            return
        self.incomplete = self.sequences_from_token_ids(self.cum_logprobs)
