import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flash_mla_sm120 import (
    FlashInferMLAWorkspace,
    _maybe_log_or_require_flashinfer_fast_path,
)


class TestFlashInferMLAWorkspace(unittest.TestCase):
    def test_reuses_decode_workspace_until_capacity_grows(self):
        workspace = FlashInferMLAWorkspace(device=torch.device("cpu"))

        first = workspace.get_decode_tensors(
            num_tokens=4,
            num_heads=64,
            num_splits=2,
            head_dim_v=512,
            output_dtype=torch.bfloat16,
        )
        second = workspace.get_decode_tensors(
            num_tokens=2,
            num_heads=64,
            num_splits=1,
            head_dim_v=512,
            output_dtype=torch.bfloat16,
        )

        self.assertEqual(first.output.data_ptr(), second.output.data_ptr())
        self.assertEqual(first.out_lse.data_ptr(), second.out_lse.data_ptr())
        self.assertEqual(first.mid_out.data_ptr(), second.mid_out.data_ptr())
        self.assertEqual(first.mid_lse.data_ptr(), second.mid_lse.data_ptr())
        self.assertEqual(second.output.shape, (2, 64, 512))
        self.assertEqual(second.mid_out.shape, (2, 64, 1, 512))

        grown = workspace.get_decode_tensors(
            num_tokens=8,
            num_heads=64,
            num_splits=2,
            head_dim_v=512,
            output_dtype=torch.bfloat16,
        )

        self.assertNotEqual(first.output.data_ptr(), grown.output.data_ptr())
        self.assertEqual(grown.output.shape, (8, 64, 512))

    def test_strict_fast_path_does_not_raise_for_prefill_shapes(self):
        with (
            envs.SGLANG_SM120_FLASHMLA_REQUIRE_DECODE_FAST_PATH.override(True),
            patch(
                "sglang.srt.layers.attention.flash_mla_sm120."
                "_flashinfer_dsv4_decode_fast_path_status",
                return_value=(False, "unsupported decode_dsv4 shape"),
            ),
        ):
            _maybe_log_or_require_flashinfer_fast_path(
                num_tokens=512,
                num_heads=64,
                d_qk=512,
                topk=128,
                page_block_size=64,
                extra_topk=0,
            )

    def test_strict_fast_path_still_raises_for_decode_shapes(self):
        with (
            envs.SGLANG_SM120_FLASHMLA_REQUIRE_DECODE_FAST_PATH.override(True),
            patch(
                "sglang.srt.layers.attention.flash_mla_sm120."
                "_flashinfer_dsv4_decode_fast_path_status",
                return_value=(False, "unsupported decode_dsv4 shape"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "fast path required"):
                _maybe_log_or_require_flashinfer_fast_path(
                    num_tokens=64,
                    num_heads=64,
                    d_qk=512,
                    topk=128,
                    page_block_size=64,
                    extra_topk=0,
                )


if __name__ == "__main__":
    unittest.main()
