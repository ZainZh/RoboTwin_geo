import unittest

import torch
import torch.nn as nn

from semantic_fusion_warmstart import (
    freeze_vanilla_backbone_for_adapter,
    load_shared_vanilla_state,
    validate_vanilla_warmstart_metadata,
)


class VanillaPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(3, 4)
        self.model = nn.Linear(4, 2)


class FusionObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = nn.Linear(3, 4)
        self.semantic_gates = nn.ParameterDict(
            {"semantic_point_cloud_A": nn.Parameter(torch.zeros(4))}
        )
        self.semantic_extractors = nn.ModuleDict(
            {"semantic_point_cloud_A": nn.Linear(5, 4)}
        )


class FusionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_encoder = FusionObsEncoder()
        self.model = nn.Linear(4, 2)


SHAPE_META_VANILLA = {
    "obs": {
        "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
        "agent_pos": {"shape": [14], "type": "low_dim"},
    },
    "action": {"shape": [14]},
}
SHAPE_META_FUSION = {
    "obs": {
        **SHAPE_META_VANILLA["obs"],
        "semantic_point_cloud_A": {"shape": [128, 131], "type": "point_cloud"},
    },
    "action": {"shape": [14]},
}


class TestSemanticFusionWarmstart(unittest.TestCase):
    def test_all_vanilla_tensors_map_into_fusion_target(self):
        torch.manual_seed(7)
        vanilla = VanillaPolicy()
        torch.manual_seed(11)
        fusion = FusionPolicy()

        source_state = dict(vanilla.state_dict())
        source_state["normalizer.params_dict.action.scale"] = torch.ones(2)
        report = load_shared_vanilla_state(fusion, source_state)

        self.assertEqual(len(vanilla.state_dict()), report["loaded_tensor_count"])
        self.assertEqual(1, report["skipped_normalizer_tensor_count"])
        self.assertTrue(
            torch.equal(vanilla.obs_encoder.weight, fusion.obs_encoder.base_encoder.weight)
        )
        self.assertTrue(torch.equal(vanilla.model.weight, fusion.model.weight))

    def test_shared_shape_mismatch_fails_closed(self):
        fusion = FusionPolicy()
        broken = dict(VanillaPolicy().state_dict())
        broken["model.weight"] = torch.randn(3, 4)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_shared_vanilla_state(fusion, broken)

    def test_freeze_keeps_only_semantic_adapter_trainable(self):
        fusion = FusionPolicy()
        report = freeze_vanilla_backbone_for_adapter(fusion)
        trainable = {
            name for name, parameter in fusion.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(
            {
                "obs_encoder.semantic_gates.semantic_point_cloud_A",
                "obs_encoder.semantic_extractors.semantic_point_cloud_A.weight",
                "obs_encoder.semantic_extractors.semantic_point_cloud_A.bias",
            },
            trainable,
        )
        self.assertGreater(report["frozen_numel"], 0)
        self.assertGreater(report["trainable_numel"], 0)

    def test_metadata_requires_strict_vanilla_matching_task_and_shapes(self):
        metadata = {
            "task_name": "beat_block_hammer-paper-sim-vanilla-joint14-e300-50",
            "pointcloud_fusion_mode": "concat",
            "use_ema": True,
            "shape_meta": SHAPE_META_VANILLA,
            "epoch": 300,
            "seed": 20260805,
        }
        report = validate_vanilla_warmstart_metadata(
            metadata,
            target_shape_meta=SHAPE_META_FUSION,
            expected_raw_task_name="beat_block_hammer",
            require_ema=True,
        )
        self.assertEqual(300, report["source_epoch"])

        with self.assertRaisesRegex(ValueError, "Strict Vanilla"):
            validate_vanilla_warmstart_metadata(
                {**metadata, "pointcloud_fusion_mode": "residual_pointnet"},
                target_shape_meta=SHAPE_META_FUSION,
                expected_raw_task_name="beat_block_hammer",
                require_ema=True,
            )
        with self.assertRaisesRegex(ValueError, "action shape mismatch"):
            validate_vanilla_warmstart_metadata(
                metadata,
                target_shape_meta={
                    **SHAPE_META_FUSION,
                    "action": {"shape": [20]},
                },
                expected_raw_task_name="beat_block_hammer",
                require_ema=True,
            )


if __name__ == "__main__":
    unittest.main()
