import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod


class TestSm120CuteDslMoeRunner(unittest.TestCase):
    def test_mxfp4_create_weights_registers_nvfp4_scales(self):
        method = Mxfp4MarlinMoEMethod(MagicMock(), prefix="model.layers.0.mlp.experts")
        layer = torch.nn.Module()
        layer.moe_runner_config = MoeRunnerConfig(activation="silu", is_gated=True, top_k=2)

        method.create_weights(
            layer,
            num_experts=4,
            hidden_size=64,
            intermediate_size_per_partition=32,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(tuple(layer.w13_weight_scale.shape), (4, 64, 4))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (4, 64, 2))
        self.assertEqual(tuple(layer.w13_weight_scale_2.shape), (4, 2))
        self.assertEqual(tuple(layer.w2_weight_scale_2.shape), (4,))
        self.assertEqual(tuple(layer.w13_input_scale.shape), (4, 2))
        self.assertEqual(tuple(layer.w2_input_scale.shape), (4,))

    def test_mxfp4_sm120_b12x_apply_passes_loaded_global_scales(self):
        method = Mxfp4MarlinMoEMethod(MagicMock(), prefix="model.layers.0.mlp.experts")
        method.runner = MagicMock()
        method.runner.config.routed_scaling_factor = None
        method.runner.run.return_value = StandardCombineInput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16)
        )

        w13_global_scale = torch.tensor([0.25, 0.5, 1.0, 2.0])
        w2_global_scale = torch.tensor([0.125, 0.25, 0.5, 1.0])
        fc2_input_scale = torch.tensor([3.0])
        layer = SimpleNamespace(
            _dsv4_mxfp4_backend="sm120_cutedsl",
            _sm120_cutedsl_packed=SimpleNamespace(
                w13=torch.empty(4, 16, 4, dtype=torch.uint8),
                w13_scale=torch.empty(1, dtype=torch.uint8),
                w13_global_scale=w13_global_scale,
                w2=torch.empty(4, 8, 8, dtype=torch.uint8),
                w2_scale=torch.empty(1, dtype=torch.uint8),
                w2_global_scale=w2_global_scale,
                fc2_input_scale=fc2_input_scale,
            ),
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 2, dtype=torch.float32),
            topk_ids=torch.zeros(2, 2, dtype=torch.int64),
            router_logits=None,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        method.apply(layer, dispatch_output)

        args, _ = method.runner.run.call_args
        self.assertIs(args[1].w1_alpha, w13_global_scale)
        self.assertIs(args[1].w2_alpha, w2_global_scale)
        self.assertIs(args[1].fc2_input_scale, fc2_input_scale)

    def test_mxfp4_sm120_b12x_create_runner_uses_flashinfer_cutedsl(self):
        method = Mxfp4MarlinMoEMethod(MagicMock(), prefix="model.layers.0.mlp.experts")
        config = MoeRunnerConfig(activation="silu", is_gated=True, top_k=2)

        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4_marlin_moe.is_sm120_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4_marlin_moe.envs.SGLANG_OPT_USE_SM120_CUTEDSL_MOE.get",
                return_value=True,
            ),
            patch("sglang.srt.layers.moe.moe_runner.MoeRunner") as moe_runner,
        ):
            method.create_moe_runner(SimpleNamespace(), config)

        moe_runner.assert_called_once_with(MoeRunnerBackend.FLASHINFER_CUTEDSL, config)

    def test_mxfp4_sm120_b12x_apply_uses_moe_runner(self):
        method = Mxfp4MarlinMoEMethod(MagicMock(), prefix="model.layers.0.mlp.experts")
        method.runner = MagicMock()
        method.runner.config.routed_scaling_factor = None
        method.runner.run.return_value = StandardCombineInput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16)
        )

        layer = SimpleNamespace(
            _dsv4_mxfp4_backend="sm120_cutedsl",
            _sm120_cutedsl_packed=SimpleNamespace(
                w13=torch.empty(4, 16, 4, dtype=torch.uint8),
                w13_scale=torch.empty(1, dtype=torch.uint8),
                w13_global_scale=torch.ones(4, dtype=torch.float32),
                w2=torch.empty(4, 8, 8, dtype=torch.uint8),
                w2_scale=torch.empty(1, dtype=torch.uint8),
                w2_global_scale=torch.ones(4, dtype=torch.float32),
                fc2_input_scale=torch.ones(1, dtype=torch.float32),
            ),
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 2, dtype=torch.float32),
            topk_ids=torch.zeros(2, 2, dtype=torch.int64),
            router_logits=None,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        output = method.apply(layer, dispatch_output)

        self.assertIs(output, method.runner.run.return_value)
        method.runner.run.assert_called_once()
        args, _ = method.runner.run.call_args
        self.assertIs(args[0], dispatch_output)
        self.assertEqual(args[1].num_experts, 4)
        self.assertEqual(args[1].top_k, 2)

    def test_auto_sm120_b12x_selects_flashinfer_cutedsl_runner(self):
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        config = MoeRunnerConfig(activation="silu", is_gated=True, top_k=2)
        layer = SimpleNamespace()

        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.is_sm120_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.envs.SGLANG_OPT_USE_SM120_CUTEDSL_MOE.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.MoeRunner"
            ) as moe_runner,
        ):
            method.create_moe_runner(layer, config)

        self.assertEqual(method._moe_runner_backend, MoeRunnerBackend.FLASHINFER_CUTEDSL)
        moe_runner.assert_called_once_with(MoeRunnerBackend.FLASHINFER_CUTEDSL, config)

    def test_sm120_b12x_apply_uses_moe_runner(self):
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method.moe_runner_config = MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            top_k=2,
            params_dtype=torch.bfloat16,
        )
        method.runner = MagicMock()
        method.runner.config.routed_scaling_factor = None
        method.runner.run.return_value = StandardCombineInput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16)
        )

        layer = SimpleNamespace(
            _nvfp4_backend="sm120_cutedsl",
            _sm120_cutedsl_nvfp4={
                "w13_weight": torch.empty(4, 16, 4, dtype=torch.uint8),
                "w13_sf": torch.empty(1, dtype=torch.uint8),
                "w13_alpha": torch.ones(4, dtype=torch.float32),
                "w2_weight": torch.empty(4, 8, 8, dtype=torch.uint8),
                "w2_sf": torch.empty(1, dtype=torch.uint8),
                "w2_alpha": torch.ones(4, dtype=torch.float32),
                "fc2_input_scale": torch.ones(1, dtype=torch.float32),
            },
            num_experts=4,
            num_local_experts=4,
        )
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 2, dtype=torch.float32),
            topk_ids=torch.zeros(2, 2, dtype=torch.int64),
            router_logits=None,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.empty(2, 8, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=topk_output,
        )

        output = method.apply(layer, dispatch_output)

        self.assertIs(output, method.runner.run.return_value)
        method.runner.run.assert_called_once()
        args, _ = method.runner.run.call_args
        self.assertIs(args[0], dispatch_output)
        self.assertEqual(args[1].num_experts, 4)
        self.assertEqual(args[1].top_k, 2)


if __name__ == "__main__":
    unittest.main()
