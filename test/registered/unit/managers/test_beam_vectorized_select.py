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
import unittest

import torch

from sglang.srt.managers.beam_vectorized_select import (
    select_final_topk,
    vectorized_select,
)
from sglang.srt.managers.beam_search_type import BeamSearchList, BeamSearchSequence
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")


class TestVectorizedSelect(unittest.TestCase):
    def test_survivors_without_stop(self):
        cum = torch.tensor([0.0, -1.0], dtype=torch.float32)
        step = torch.tensor([[-0.1, -0.5, -1.0], [-0.2, -0.6, -1.2]], dtype=torch.float32)
        toks = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        stop = torch.tensor([], dtype=torch.int64)

        sel = vectorized_select(cum, step, toks, stop, beam_width=2)

        self.assertEqual(int(sel.num_survivors), 2)
        self.assertEqual(int(sel.num_finished), 0)
        self.assertEqual(sel.next_tokens.shape, (2,))
        self.assertEqual(sel.parent_idx.shape, (2,))
        # Best: row0+tok10 (-0.1), then row0+tok11 (-0.5) or row1+tok20 (-1.2)?
        # scores: [-0.1, -0.5, -1.0, -1.2, -1.6, -2.2]
        self.assertEqual(sel.next_tokens[:2].tolist(), [10, 11])
        self.assertEqual(sel.parent_idx[:2].tolist(), [0, 0])

    def test_eos_finishes_and_examined_truncation(self):
        cum = torch.tensor([0.0, 0.0], dtype=torch.float32)
        # 4 candidates so ranking can see both EOS and both survivors.
        step = torch.tensor(
            [[-0.1, -0.2, -1.0, -1.1], [-0.3, -0.4, -1.2, -1.3]], dtype=torch.float32
        )
        toks = torch.tensor(
            [[100, 101, 110, 111], [102, 103, 112, 113]], dtype=torch.int64
        )
        stop = torch.tensor([100, 102], dtype=torch.int64)

        sel = vectorized_select(cum, step, toks, stop, beam_width=2)

        self.assertEqual(int(sel.num_survivors), 2)
        self.assertEqual(sel.next_tokens[:2].tolist(), [101, 103])
        self.assertEqual(sel.parent_idx[:2].tolist(), [0, 1])
        # Order: 100 eos, 101 surv#1, 102 eos, 103 surv#2 -> both eos examined.
        self.assertEqual(int(sel.num_finished), 2)
        self.assertEqual(sel.fin_tokens[:2].tolist(), [100, 102])

    def test_examined_stops_after_k_survivors(self):
        cum = torch.tensor([0.0], dtype=torch.float32)
        # beam_width=1: after first non-stop, later EOS must not finish.
        step = torch.tensor([[-0.1, -0.2, -0.3]], dtype=torch.float32)
        toks = torch.tensor([[10, 99, 11]], dtype=torch.int64)
        stop = torch.tensor([99], dtype=torch.int64)

        sel = vectorized_select(cum, step, toks, stop, beam_width=1)

        self.assertEqual(int(sel.num_survivors), 1)
        self.assertEqual(int(sel.next_tokens[0]), 10)
        self.assertEqual(int(sel.num_finished), 0)

    def test_fixed_output_shapes(self):
        cum = torch.zeros(3, dtype=torch.float32)
        step = torch.randn(3, 5)
        toks = torch.randint(0, 50, (3, 5), dtype=torch.int64)
        stop = torch.tensor([7], dtype=torch.int64)
        sel = vectorized_select(cum, step, toks, stop, beam_width=3)
        self.assertEqual(sel.next_tokens.shape, (3,))
        self.assertEqual(sel.parent_idx.shape, (3,))
        self.assertEqual(sel.new_cum_logprobs.shape, (3,))
        self.assertEqual(sel.fin_tokens.shape, (5,))
        self.assertEqual(sel.fin_parent_idx.shape, (5,))
        self.assertEqual(sel.fin_cum_logprobs.shape, (5,))
        self.assertEqual(sel.num_survivors.shape, ())
        self.assertEqual(sel.num_finished.shape, ())

    def test_select_final_topk(self):
        cum = torch.tensor([0.0, -1.0], dtype=torch.float32)
        step = torch.tensor([[-0.1, -0.5], [-0.2, -0.6]], dtype=torch.float32)
        toks = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        final = select_final_topk(cum, step, toks, beam_width=2)
        self.assertEqual(final.tokens.shape, (2,))
        self.assertEqual(final.parent_idx.shape, (2,))
        self.assertEqual(final.cum_logprobs.shape, (2,))
        self.assertEqual(final.tokens.tolist(), [1, 2])


class TestBeamSearchListDense(unittest.TestCase):
    def test_expand_token_ids_gather_append(self):
        bl = BeamSearchList()
        bl.incomplete = [
            BeamSearchSequence(tokens=[1, 2], cum_logprob=-1.0),
            BeamSearchSequence(tokens=[3, 4], cum_logprob=-2.0),
        ]
        bl.init_token_ids(max_new_tokens=8, device=torch.device("cpu"))
        self.assertEqual(bl.cur_len, 2)
        self.assertTrue(bl.dense_authoritative)

        bl.expand_token_ids([1, 0], [9, 8])
        self.assertEqual(bl.cur_len, 3)
        self.assertEqual(bl.token_ids[0, :3].tolist(), [3, 4, 9])
        self.assertEqual(bl.token_ids[1, :3].tolist(), [1, 2, 8])

        seqs = bl.sequences_from_token_ids([-2.5, -1.5])
        self.assertEqual(seqs[0].tokens, [3, 4, 9])
        self.assertEqual(seqs[1].tokens, [1, 2, 8])

    def test_expand_clamps_oob_parent(self):
        bl = BeamSearchList()
        bl.incomplete = [BeamSearchSequence(tokens=[5], cum_logprob=0.0)]
        bl.init_token_ids(max_new_tokens=4, device=torch.device("cpu"))
        bl.expand_token_ids([99], [7])
        self.assertEqual(bl.token_ids[0, :2].tolist(), [5, 7])

    def test_ensure_empty_stubs_reuse(self):
        bl = BeamSearchList()
        stubs = bl.ensure_empty_stubs(2)
        self.assertEqual(len(stubs), 2)
        self.assertEqual(stubs[0].tokens, [])
        bl.incomplete = stubs
        again = bl.ensure_empty_stubs(2)
        self.assertIs(again, stubs)


if __name__ == "__main__":
    unittest.main()
