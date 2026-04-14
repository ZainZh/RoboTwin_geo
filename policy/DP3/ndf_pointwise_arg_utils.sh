#!/bin/bash

looks_like_placeholder_arg() {
    local value="${1:-}"
    [[ "${value}" == *"{"*"}"* ]]
}

looks_like_integer_arg() {
    local value="${1:-}"
    [[ "${value}" =~ ^[0-9]+$ ]]
}

normalize_ndf_train_args() {
    local args=("$@")
    task_name="${args[0]}"
    task_config="${args[1]}"
    expert_data_num="${args[2]}"
    seed="${args[3]}"
    gpu_id="${args[4]}"
    ndf_ckpt_A="${args[5]:-none}"
    ndf_ckpt_B="${args[6]:-none}"
    ndf_device="${args[7]:-cuda:0}"
    ndf_dgcnn_placeholders="${args[8]:-}"
    object_placeholders="${args[9]:-\{A\},\{B\}}"
    ndf_point_num="${args[10]:-128}"
    ndf_legacy_shifted=false

    if looks_like_placeholder_arg "${args[8]:-}" && looks_like_integer_arg "${args[9]:-}"; then
        ndf_dgcnn_placeholders=""
        object_placeholders="${args[8]}"
        ndf_point_num="${args[9]}"
        ndf_legacy_shifted=true
    fi
}

normalize_ndf_eval_args() {
    local args=("$@")
    policy_name=DP3
    task_name="${args[0]}"
    task_config="${args[1]}"
    ckpt_setting="${args[2]}-objpc-ndf-pointwise"
    expert_data_num="${args[3]}"
    seed="${args[4]}"
    gpu_id="${args[5]}"
    ndf_ckpt_A="${args[6]:-none}"
    ndf_ckpt_B="${args[7]:-none}"
    ndf_device="${args[8]:-cuda:0}"
    ndf_dgcnn_placeholders="${args[9]:-}"
    object_placeholders="${args[10]:-\{A\},\{B\}}"
    checkpoint_num="${args[11]:-3000}"
    ndf_point_num="${args[12]:-128}"
    ndf_legacy_shifted=false

    if looks_like_placeholder_arg "${args[9]:-}" && looks_like_integer_arg "${args[10]:-}"; then
        ndf_dgcnn_placeholders=""
        object_placeholders="${args[9]}"
        checkpoint_num="${args[10]}"
        ndf_point_num="${args[11]:-128}"
        ndf_legacy_shifted=true
    fi
}

normalize_ndf_eval_hybrid_args() {
    local args=("$@")
    policy_name=DP3
    task_name="${args[0]}"
    task_config="${args[1]}"
    expert_data_num="${args[2]}"
    seed="${args[3]}"
    gpu_id="${args[4]}"
    ndf_ckpt_A="${args[5]:-none}"
    ndf_ckpt_B="${args[6]:-none}"
    ndf_device="${args[7]:-cuda:0}"
    ndf_dgcnn_placeholders="${args[8]:-}"
    object_placeholders="${args[9]:-\{A\},\{B\}}"
    checkpoint_num="${args[10]:-3000}"
    ndf_point_num="${args[11]:-128}"
    ndf_legacy_shifted=false

    if looks_like_placeholder_arg "${args[8]:-}" && looks_like_integer_arg "${args[9]:-}"; then
        ndf_dgcnn_placeholders=""
        object_placeholders="${args[8]}"
        checkpoint_num="${args[9]}"
        ndf_point_num="${args[10]:-128}"
        ndf_legacy_shifted=true
    fi
}

normalize_ndf_process_actorseg_args() {
    local args=("$@")
    task_name="${args[0]}"
    task_config="${args[1]}"
    expert_data_num="${args[2]}"
    ndf_ckpt_A="${args[3]:-none}"
    ndf_ckpt_B="${args[4]:-none}"
    ndf_device="${args[5]:-cuda:0}"
    ndf_dgcnn_placeholders="${args[6]:-}"
    object_placeholders="${args[7]:-\{A\},\{B\}}"
    ndf_point_num="${args[8]:-128}"
    actorseg_camera_names="${args[9]:-head_camera,front_camera}"
    ndf_legacy_shifted=false

    if looks_like_placeholder_arg "${args[6]:-}" && looks_like_integer_arg "${args[7]:-}"; then
        ndf_dgcnn_placeholders=""
        object_placeholders="${args[6]}"
        ndf_point_num="${args[7]}"
        actorseg_camera_names="${args[8]:-head_camera,front_camera}"
        ndf_legacy_shifted=true
    fi
}
