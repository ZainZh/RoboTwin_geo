from __future__ import annotations

import torch

from .diffusion_action_decoder import (
    DiffusionActionDecoder,
    stateless_normal_like,
)
from .interaction_flow_tokens import (
    InteractionFlowTokenPolicy,
    PointTokenActionPolicy,
    attention_entropy,
    correspondence_flow_loss,
    motion_correspondence_flow_loss,
    rigid_flow_loss,
    unordered_flow_chamfer_loss,
)
from .train_interaction_flow_tokens import (
    _rotation_matrix_from_rotvec,
    active_endpoint_rotation_geodesic_loss,
    initialize_recovery_adapter_from_decoder,
)


def batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "points_a": torch.randn(batch_size, 24, 3, generator=generator),
        "points_b": torch.randn(batch_size, 20, 3, generator=generator),
        "state": torch.randn(batch_size, 20, generator=generator),
    }


def test_rotvec_exponential_map_is_identity_and_has_finite_gradient() -> None:
    rotation_vector = torch.zeros(4, 3, requires_grad=True)
    matrix = _rotation_matrix_from_rotvec(rotation_vector)
    expected = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(matrix, expected, atol=1e-7)
    matrix.square().sum().backward()
    assert rotation_vector.grad is not None
    assert torch.isfinite(rotation_vector.grad).all()


def test_endpoint_rotation_loss_uses_active_arm_and_physical_units() -> None:
    prediction = torch.zeros(2, 2, 14, requires_grad=True)
    target = torch.zeros_like(prediction)
    with torch.no_grad():
        prediction[0, 0, 5] = 0.2  # left-arm z rotvec
        prediction[1, 0, 12] = 0.4  # right-arm z rotvec
        prediction[1, 0, 5] = 1.0  # inactive left arm must be ignored
    mean = torch.zeros(14)
    std = torch.ones(14)
    std[[5, 12]] = 0.5
    loss = active_endpoint_rotation_geodesic_loss(
        prediction,
        target,
        torch.tensor([0.0, 1.0]),
        mean,
        std,
    )
    # Both active-arm physical rotations are 0.1 and 0.2 rad.
    assert torch.allclose(loss, torch.tensor(0.15), atol=2e-5)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(prediction.grad[1, 0, 5].abs()) == 0.0


def test_endpoint_rotation_loss_composes_the_chunk_not_individual_steps() -> None:
    prediction = torch.zeros(1, 2, 14)
    target = torch.zeros_like(prediction)
    # The per-step targets differ, but both chunks end at +0.2 rad about z.
    prediction[0, 1, 5] = 0.2
    target[0, :, 5] = 0.1
    loss = active_endpoint_rotation_geodesic_loss(
        prediction,
        target,
        torch.tensor([0.0]),
        torch.zeros(14),
        torch.ones(14),
    )
    assert float(loss) < 2e-5


def test_decoder_cloned_recovery_adapter_is_aligned_and_exact_zero() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        recovery_action_adapter=True,
    )
    with torch.no_grad():
        model.decoder.action_queries.fill_(0.37)
        model.decoder.action_head.weight.fill_(0.21)
        model.decoder.action_head.bias.fill_(-0.14)
        model.recovery_action_adapter.action_queries.fill_(-0.9)
    initialize_recovery_adapter_from_decoder(model)
    torch.testing.assert_close(
        model.recovery_action_adapter.action_queries,
        model.decoder.action_queries,
    )
    assert torch.equal(
        model.recovery_action_adapter.action_head.weight,
        torch.zeros_like(model.recovery_action_adapter.action_head.weight),
    )
    assert torch.equal(
        model.recovery_action_adapter.action_head.bias,
        torch.zeros_like(model.recovery_action_adapter.action_head.bias),
    )
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    value["target_frame_confidence"] = torch.ones(3)
    output = model(value)
    assert torch.equal(
        output.recovery_action_residual,
        torch.zeros_like(output.recovery_action_residual),
    )


def test_point_token_policy_shape() -> None:
    model = PointTokenActionPolicy(
        horizon=6,
        feature_dim=32,
        heads=4,
        layers=1,
    )
    output = model(batch())
    assert output.action.shape == (3, 6, 14)
    assert output.predicted_flow is None


def test_source_geometry_adapter_is_noop_then_receives_gradient() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        point_channels=6,
        source_geometry_adapter=True,
    )
    model = InteractionFlowTokenPolicy(**arguments).eval()
    first = batch()
    first["points_a"] = torch.cat(
        (first["points_a"], torch.randn(3, 24, 3)), dim=-1
    )
    first["points_b"] = torch.cat(
        (first["points_b"], torch.zeros(3, 20, 3)), dim=-1
    )
    second = {key: value.clone() for key, value in first.items()}
    second["points_a"][..., 3:] = torch.randn_like(second["points_a"][..., 3:])
    first_action = model(first).action
    second_action = model(second).action
    assert torch.equal(first_action, second_action)
    first_action.square().mean().backward()
    assert model.source_geometry_projection.gate.grad is not None
    assert torch.isfinite(model.source_geometry_projection.gate.grad)
    with torch.no_grad():
        model.source_geometry_projection.gate.fill_(0.25)
    assert not torch.allclose(model(first).action, model(second).action)


def test_source_geometry_and_target_color_adapters_are_exclusive() -> None:
    try:
        InteractionFlowTokenPolicy(
            horizon=3,
            flow_steps=4,
            num_anchors=6,
            feature_dim=32,
            heads=4,
            layers=1,
            point_channels=6,
            target_color_adapter=True,
            source_geometry_adapter=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("expected mutually-exclusive adapters to fail")


def test_zero_source_descriptor_cannot_become_a_constant_residual() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        point_channels=6,
        source_geometry_adapter=True,
    )
    descriptor = torch.zeros(2, 11, 3)
    with torch.no_grad():
        model.source_geometry_projection.gate.fill_(0.7)
    feature = model.source_geometry_projection(descriptor)
    assert torch.equal(feature, torch.zeros_like(feature))


def test_source_axis_relation_adapter_is_gated_and_zero_safe() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        point_channels=6,
        source_axis_relation_adapter=True,
    ).eval()
    value = batch()
    value["points_a"] = torch.cat((value["points_a"], torch.randn(3, 24, 3)), -1)
    value["points_b"] = torch.cat((value["points_b"], torch.randn(3, 20, 3)), -1)
    value["source_axis3"] = torch.randn(3, 3)
    changed = {key: tensor.clone() for key, tensor in value.items()}
    changed["source_axis3"] = torch.randn(3, 3)
    original_action = model(value).action
    assert torch.equal(original_action, model(changed).action)
    original_action.square().mean().backward()
    adapter = model.source_axis_relation_projection
    assert adapter.gate.grad is not None and torch.isfinite(adapter.gate.grad)
    with torch.no_grad():
        adapter.gate.fill_(0.4)
    assert not torch.allclose(model(value).action, model(changed).action)
    zero_axis = torch.zeros(3, 3)
    delta = torch.randn(3, 6, 3)
    assert torch.equal(
        adapter(zero_axis, delta, model.xyz_std),
        torch.zeros(3, 6, 32),
    )


def test_target_axis_relation_adapter_is_incremental_and_zero_safe() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        source_axis_relation_adapter=True,
        target_axis_relation_adapter=True,
    ).eval()
    value = batch()
    value["source_axis3"] = torch.randn(3, 3)
    value["target_axis3"] = torch.randn(3, 3)
    changed = {key: tensor.clone() for key, tensor in value.items()}
    changed["target_axis3"] = torch.randn(3, 3)
    original_action = model(value).action
    assert torch.equal(original_action, model(changed).action)
    original_action.square().mean().backward()
    adapter = model.target_axis_relation_projection
    assert adapter.gate.grad is not None and torch.isfinite(adapter.gate.grad)
    with torch.no_grad():
        adapter.gate.fill_(0.4)
    assert not torch.allclose(model(value).action, model(changed).action)
    zeros = torch.zeros(3, 3)
    delta = torch.randn(3, 6, 3)
    assert torch.equal(
        adapter(value["source_axis3"], zeros, delta, model.xyz_std),
        torch.zeros(3, 6, 32),
    )


def test_interaction_policy_shapes_and_gradients() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=6,
        flow_steps=4,
        num_anchors=8,
        feature_dim=32,
        heads=4,
        layers=1,
    )
    output = model(batch())
    assert output.action.shape == (3, 6, 14)
    assert output.anchor_a.shape == (3, 8, 3)
    assert output.anchor_b.shape == (3, 8, 3)
    assert output.predicted_flow.shape == (3, 8, 4, 3)
    assert output.attention_a.shape == (3, 8, 24)
    assert output.attention_b.shape == (3, 8, 20)
    assert torch.allclose(
        output.predicted_flow[:, :, 0], output.anchor_a, atol=1e-6
    )
    loss = output.action.square().mean() + output.predicted_flow.square().mean()
    loss.backward()
    assert model.anchor_queries.grad is not None
    assert torch.isfinite(model.anchor_queries.grad).all()


def test_flow_conditioned_action_backpropagates_into_flow_head() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        condition_action_on_flow=True,
    )
    # Make the residual path nonzero; it is deliberately zero-initialized for
    # stable training and becomes active after the first optimizer update.
    torch.nn.init.normal_(model.flow_condition_projection[-1].weight, std=0.01)
    output = model(batch())
    output.action.square().mean().backward()
    assert model.flow_head[-1].weight.grad is not None
    assert float(model.flow_head[-1].weight.grad.abs().sum()) > 0.0


def test_functional_frame_is_an_explicit_action_memory_token() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
    )
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    output = model(value)
    output.action.square().mean().backward()
    assert model.functional_frame_projection[0].weight.grad is not None
    assert output.action.shape == (3, 3, 14)


def test_functional_frame_confidence_masks_projection_bias() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
    ).eval()
    first = batch()
    first["target_frame9"] = torch.randn(3, 9)
    first["target_frame_confidence"] = torch.zeros(3)
    second = {key: value.clone() for key, value in first.items()}
    second["target_frame9"] = torch.randn(3, 9)
    assert torch.equal(model(first).action, model(second).action)


def test_gated_functional_frame_starts_as_exact_residual_noop() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        functional_frame_gated_scene=True,
        functional_frame_as_memory=False,
    ).eval()
    first = batch()
    first["target_frame9"] = torch.randn(3, 9)
    second = {key: value.clone() for key, value in first.items()}
    second["target_frame9"] = torch.randn(3, 9)
    first_action = model(first).action
    second_action = model(second).action
    assert torch.equal(first_action, second_action)
    first_action.square().mean().backward()
    assert model.functional_frame_gate.grad is not None


def test_dual_frame_gate_zero_exactly_matches_single_frame_policy() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        functional_frame_as_memory=True,
    )
    torch.manual_seed(41)
    single = InteractionFlowTokenPolicy(**arguments).eval()
    torch.manual_seed(41)
    dual = InteractionFlowTokenPolicy(
        **arguments, dual_functional_frame_token=True
    ).eval()
    incompatible = dual.load_state_dict(single.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "dual_functional_frame_gate",
        "dual_functional_frame_projection.0.weight",
        "dual_functional_frame_projection.0.bias",
        "dual_functional_frame_projection.1.weight",
        "dual_functional_frame_projection.1.bias",
        "dual_functional_frame_projection.3.weight",
        "dual_functional_frame_projection.3.bias",
    }
    assert not incompatible.unexpected_keys
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    value["target_frame9_local"] = torch.randn(3, 9)
    value["target_frame_confidence"] = torch.ones(3)
    with torch.no_grad():
        single_output = single(value)
        dual_output = dual(value)
    torch.testing.assert_close(single_output.action, dual_output.action)
    torch.testing.assert_close(
        single_output.predicted_flow, dual_output.predicted_flow
    )


def test_recovery_action_adapter_starts_as_exact_policy_noop() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
    )
    baseline = InteractionFlowTokenPolicy(**arguments).eval()
    adapted = InteractionFlowTokenPolicy(
        **arguments, recovery_action_adapter=True
    ).eval()
    incompatible = adapted.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        name.startswith("recovery_action_adapter.")
        for name in incompatible.missing_keys
    )
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    value["target_frame_confidence"] = torch.ones(3)
    with torch.no_grad():
        baseline_action = baseline(value).action
        adapted_output = adapted(value)
        adapted_action = adapted_output.action
    torch.testing.assert_close(baseline_action, adapted_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        adapted_output.recovery_action_residual,
        torch.zeros_like(adapted_output.recovery_action_residual),
        rtol=0.0,
        atol=0.0,
    )


def test_recovery_action_adapter_is_exactly_off_near_goal() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
    )
    baseline = InteractionFlowTokenPolicy(**arguments).eval()
    adapted = InteractionFlowTokenPolicy(
        **arguments, recovery_action_adapter=True
    ).eval()
    adapted.load_state_dict(baseline.state_dict(), strict=False)
    torch.nn.init.normal_(adapted.recovery_action_adapter.action_head.weight)
    value = batch()
    identity = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    value["target_frame9"] = identity.repeat(3, 1)
    value["target_frame_confidence"] = torch.ones(3)
    with torch.no_grad():
        baseline_action = baseline(value).action
        adapted_action = adapted(value).action
    torch.testing.assert_close(baseline_action, adapted_action, rtol=0.0, atol=0.0)


def test_source_functional_recovery_rotates_metric_vectors_equivariantly() -> None:
    action_std = [
        1.0,
        2.0,
        4.0,
        3.0,
        6.0,
        12.0,
        1.0,
        2.0,
        5.0,
        8.0,
        4.0,
        7.0,
        10.0,
        1.0,
    ]
    model = InteractionFlowTokenPolicy(
        horizon=2,
        flow_steps=3,
        num_anchors=4,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        recovery_action_adapter=True,
        recovery_action_frame="source_functional",
        action_std=action_std,
    )
    residual = torch.zeros(1, 2, 14)
    residual[..., 0] = 1.0
    residual[..., 4] = -0.5
    residual[..., 9] = 0.25
    residual[..., 12] = 0.75
    residual[..., 6] = 0.3
    residual[..., 13] = -0.2
    # Local X maps to world Y under +90 degrees around world Z.
    source_frame6 = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])
    result = model._recovery_residual_in_world_frame(
        residual, {"source_frame6_columns": source_frame6}
    )
    rotation = model._rotation_sixd(source_frame6)[0]
    std = torch.tensor(action_std)
    for start in (0, 3, 7, 10):
        local_metric = residual[0, :, start : start + 3] * std[
            start : start + 3
        ].mean()
        expected_metric = torch.einsum("ij,hj->hi", rotation, local_metric)
        actual_metric = result[0, :, start : start + 3] * std[
            start : start + 3
        ]
        torch.testing.assert_close(actual_metric, expected_metric)
    torch.testing.assert_close(result[..., 6], residual[..., 6])
    torch.testing.assert_close(result[..., 13], residual[..., 13])

    hybrid = InteractionFlowTokenPolicy(
        horizon=2,
        flow_steps=3,
        num_anchors=4,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        recovery_action_adapter=True,
        recovery_action_frame="hybrid_source_rotation",
        recovery_local_frame_token=True,
        action_std=action_std,
    )
    hybrid_result = hybrid._recovery_residual_in_world_frame(
        residual, {"source_frame6_columns": source_frame6}
    )
    torch.testing.assert_close(hybrid_result[..., :3], residual[..., :3])
    torch.testing.assert_close(hybrid_result[..., 7:10], residual[..., 7:10])
    for start in (3, 10):
        local_metric = residual[0, :, start : start + 3] * std[
            start : start + 3
        ].mean()
        expected_metric = torch.einsum("ij,hj->hi", rotation, local_metric)
        actual_metric = hybrid_result[0, :, start : start + 3] * std[
            start : start + 3
        ]
        torch.testing.assert_close(actual_metric, expected_metric)


def test_source_functional_recovery_starts_as_exact_policy_noop() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        action_std=[float(value) for value in range(1, 15)],
    )
    baseline = InteractionFlowTokenPolicy(**arguments).eval()
    adapted = InteractionFlowTokenPolicy(
        **arguments,
        recovery_action_adapter=True,
        recovery_action_frame="source_functional",
        recovery_local_frame_token=True,
    ).eval()
    incompatible = adapted.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith(
            ("recovery_action_adapter.", "recovery_local_frame_projection.")
        )
        for name in incompatible.missing_keys
    )
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    value["target_frame_confidence"] = torch.ones(3)
    identity6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    value["source_frame6_columns"] = identity6.repeat(3, 1)
    value["target_frame9_source_local"] = torch.randn(3, 9)
    with torch.no_grad():
        baseline_action = baseline(value).action
        adapted_output = adapted(value)
    torch.testing.assert_close(
        baseline_action, adapted_output.action, rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(adapted_output.recovery_action_residual) == 0
    torch.nn.init.normal_(adapted.recovery_action_adapter.action_head.weight)
    adapted.zero_grad(set_to_none=True)
    adapted(value).action.square().mean().backward()
    assert adapted.recovery_local_frame_projection[0].weight.grad is not None
    assert torch.isfinite(
        adapted.recovery_local_frame_projection[0].weight.grad
    ).all()


def test_gated_biframe_is_exact_world_adapter_at_zero_gate() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        recovery_action_adapter=True,
    )
    torch.manual_seed(71)
    world = InteractionFlowTokenPolicy(**arguments).eval()
    torch.nn.init.normal_(world.recovery_action_adapter.action_head.weight)
    torch.nn.init.normal_(world.recovery_action_adapter.action_head.bias)
    torch.manual_seed(83)
    gated = InteractionFlowTokenPolicy(
        **arguments,
        recovery_local_frame_token=True,
        recovery_local_frame_gated_fusion=True,
    ).eval()
    incompatible = gated.load_state_dict(world.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "recovery_local_frame_gate",
        "recovery_local_frame_projection.0.weight",
        "recovery_local_frame_projection.0.bias",
        "recovery_local_frame_projection.1.weight",
        "recovery_local_frame_projection.1.bias",
        "recovery_local_frame_projection.3.weight",
        "recovery_local_frame_projection.3.bias",
    }
    value = batch()
    value["target_frame9"] = torch.randn(3, 9)
    value["target_frame_confidence"] = torch.ones(3)
    value["target_frame9_source_local"] = torch.randn(3, 9)
    with torch.no_grad():
        world_output = world(value)
        gated_output = gated(value)
    torch.testing.assert_close(
        gated_output.action, world_output.action, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        gated_output.recovery_action_residual,
        world_output.recovery_action_residual,
        rtol=0.0,
        atol=0.0,
    )


def test_relative_flow_tokens_start_as_gated_residual() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        functional_frame_gated_scene=True,
        functional_frame_as_memory=False,
        functional_relative_flow=True,
    ).eval()
    first = batch()
    identity = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    first["target_frame9"] = identity[None].expand(3, -1).clone()
    second = {key: value.clone() for key, value in first.items()}
    second["target_frame9"][:, 0] = 1.0
    first_action = model(first).action
    second_action = model(second).action
    assert torch.equal(first_action, second_action)
    first_action.square().mean().backward()
    assert model.functional_relative_gate.grad is not None
    with torch.no_grad():
        model.functional_relative_gate.fill_(0.25)
    shifted_action = model(second).action
    assert not torch.allclose(first_action, shifted_action)


def test_optional_relative_adapter_preserves_shared_initialization() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        predict_rotation=True,
    )
    torch.manual_seed(19)
    baseline = InteractionFlowTokenPolicy(**arguments)
    torch.manual_seed(19)
    relative = InteractionFlowTokenPolicy(
        **arguments,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        functional_frame_gated_scene=True,
        functional_frame_as_memory=False,
        functional_relative_flow=True,
    )
    baseline_state = baseline.state_dict()
    relative_state = relative.state_dict()
    for name in sorted(set(baseline_state) & set(relative_state)):
        assert torch.equal(baseline_state[name], relative_state[name]), name


def test_diffusion_decoder_training_and_sampling_are_finite() -> None:
    decoder = DiffusionActionDecoder(
        horizon=3,
        feature_dim=32,
        heads=4,
        layers=1,
        dropout=0.0,
        diffusion_steps=20,
        inference_steps=4,
    ).eval()
    clean = torch.randn(2, 3, 14)
    index = torch.tensor([4, 9])
    noisy, timestep, target = decoder.training_inputs(
        clean, sample_index=index
    )
    scene = torch.randn(2, 32)
    memory = torch.randn(2, 6, 32)
    prediction = decoder.predict_noise(noisy, timestep, scene, memory)
    prediction.square().mean().backward()
    assert decoder.prediction_head.weight.grad is not None
    initial = stateless_normal_like(clean, index, salt=97.0)
    first = decoder.sample(scene, memory, initial_noise=initial)
    second = decoder.sample(scene, memory, initial_noise=initial)
    assert first.shape == clean.shape
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert target.shape == clean.shape


def test_relative_diffusion_preserves_shared_initialization_and_noop() -> None:
    arguments = dict(
        horizon=3,
        flow_steps=4,
        num_anchors=6,
        feature_dim=32,
        heads=4,
        layers=1,
        predict_rotation=True,
        action_decoder_type="diffusion",
        diffusion_steps=20,
        diffusion_inference_steps=4,
    )
    torch.manual_seed(23)
    baseline = InteractionFlowTokenPolicy(**arguments).eval()
    torch.manual_seed(23)
    relative = InteractionFlowTokenPolicy(
        **arguments,
        functional_frame_token=True,
        functional_frame_in_scene=True,
        functional_frame_gated_scene=True,
        functional_frame_as_memory=False,
        functional_relative_flow=True,
    ).eval()
    baseline_state = baseline.state_dict()
    relative_state = relative.state_dict()
    for name in sorted(set(baseline_state) & set(relative_state)):
        assert torch.equal(baseline_state[name], relative_state[name]), name
    value = batch()
    value["target_frame9"] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )[None].expand(3, -1).clone()
    noisy = torch.randn(3, 3, 14)
    timestep = torch.tensor([2, 7, 11])
    baseline_batch = dict(value)
    baseline_batch["diffusion_noisy_action"] = noisy
    baseline_batch["diffusion_timestep"] = timestep
    baseline_output = baseline(baseline_batch).action
    relative_output = relative(baseline_batch).action
    assert torch.equal(baseline_output, relative_output)


def test_rigid_se3_parameterization_preserves_metric_distances() -> None:
    model = InteractionFlowTokenPolicy(
        horizon=6,
        flow_steps=4,
        num_anchors=8,
        feature_dim=32,
        heads=4,
        layers=1,
        flow_parameterization="rigid_se3",
        xyz_std=[0.2, 0.1, 0.05],
    )
    output = model(batch())
    assert output.predicted_flow.shape == (3, 8, 4, 3)
    scale = model.xyz_std.reshape(1, 1, 1, 3)
    metric_flow = output.predicted_flow * scale
    distance = torch.cdist(
        metric_flow.permute(0, 2, 1, 3),
        metric_flow.permute(0, 2, 1, 3),
    )
    assert torch.allclose(distance, distance[:, :1], atol=1e-5)


def test_chamfer_is_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(3)
    target = torch.randn(2, 7, 4, 3, generator=generator)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    loss = unordered_flow_chamfer_loss(target, target[:, permutation])
    assert float(loss) < 1e-7


def test_rigidity_loss_distinguishes_nonrigid_motion() -> None:
    generator = torch.Generator().manual_seed(11)
    anchors = torch.randn(2, 6, 3, generator=generator)
    translation = torch.randn(2, 1, 4, 3, generator=generator)
    rigid = anchors[:, :, None, :] + translation
    nonrigid = rigid.clone()
    nonrigid[:, 0, -1, 0] += 0.5
    assert float(rigid_flow_loss(rigid)) < 1e-8
    assert float(rigid_flow_loss(nonrigid)) > float(rigid_flow_loss(rigid)) + 1e-4


def test_correspondence_loss_rejects_temporal_anchor_swaps() -> None:
    target = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
          [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]]]
    )
    swapped = target.clone()
    swapped[:, 0, 1] = target[:, 1, 1]
    swapped[:, 1, 1] = target[:, 0, 1]
    assert float(unordered_flow_chamfer_loss(swapped, target)) < 1e-8
    assert float(correspondence_flow_loss(swapped, target)) > 0.1
    assert float(motion_correspondence_flow_loss(swapped, target)) > 1.0


def test_attention_entropy_range() -> None:
    uniform = torch.full((2, 3, 5), 0.2)
    one_hot = torch.zeros(2, 3, 5)
    one_hot[..., 0] = 1.0
    assert torch.allclose(attention_entropy(uniform), torch.tensor(1.0), atol=1e-6)
    assert float(attention_entropy(one_hot)) < 1e-5
