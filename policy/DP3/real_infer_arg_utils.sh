#!/bin/bash

resolve_real_zed_output_frame() {
    local repo_root=${1}
    local task_name=${2}
    local task_config=${3}
    local requested=${4:-auto}

    if [ -n "${requested}" ] && [ "${requested}" != "auto" ]; then
        printf "%s" "${requested}"
        return
    fi

    local meta_path="${repo_root}/data/${task_name}/${task_config}/real_zed_sam2_objpc_meta.json"
    if [ ! -f "${meta_path}" ]; then
        infer_real_zed_output_frame_from_task_config "${task_config}"
        return
    fi

    python3 - "${meta_path}" <<'PY'
import json
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

frame = str(meta.get("output_frame") or "source").strip()
allowed = {"source", "workspace", "left_base", "right_base"}
if frame not in allowed:
    raise SystemExit(f"Invalid output_frame={frame!r} in {meta_path}; expected one of {sorted(allowed)}")
print(frame, end="")
PY
}

is_real_zed_output_frame_shorthand() {
    case "${1:-}" in
        --source|--workspace|--left_base|--leftbase|--right_base|--rightbase)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_real_zed_output_frame_token() {
    case "${1:-}" in
        --source)
            printf "source"
            ;;
        --workspace)
            printf "workspace"
            ;;
        --left_base|--leftbase)
            printf "left_base"
            ;;
        --right_base|--rightbase)
            printf "right_base"
            ;;
        *)
            printf "%s" "${1:-auto}"
            ;;
    esac
}

normalize_real_zed_calibration_token() {
    case "${1:-}" in
        --auto)
            printf "auto"
            ;;
        --none)
            printf "none"
            ;;
        *)
            printf "%s" "${1:-auto}"
            ;;
    esac
}

infer_real_zed_output_frame_from_task_config() {
    local task_config=${1}
    case "${task_config}" in
        *right_base*|*rightbase*)
            printf "right_base"
            ;;
        *left_base*|*leftbase*)
            printf "left_base"
            ;;
        *workspace*)
            printf "workspace"
            ;;
        *)
            printf "source"
            ;;
    esac
}

resolve_real_zed_robot_camera_calibration_path() {
    local repo_root=${1}
    local output_frame=${2}
    local requested=${3:-auto}

    if [ -n "${requested}" ] && [ "${requested}" != "auto" ] && [ "${requested}" != "none" ]; then
        printf "%s" "${requested}"
        return
    fi

    if [ "${requested}" = "none" ]; then
        return
    fi

    case "${output_frame}" in
        left_base)
            printf "%s" "${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml"
            ;;
        right_base)
            printf "%s" "${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml"
            ;;
        *)
            return
            ;;
    esac
}
