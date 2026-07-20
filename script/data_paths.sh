#!/bin/bash

_robotwin_data_paths_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTWIN_REPO_ROOT=$(cd "${_robotwin_data_paths_dir}/.." && pwd)
ROBOTWIN_SERVER_STORAGE_ROOT=${ROBOTWIN_SERVER_STORAGE_ROOT:-/shared2/sz}

if [ -d "${ROBOTWIN_SERVER_STORAGE_ROOT}" ]; then
    ROBOTWIN_STORAGE_MODE=server
    _robotwin_storage_repo="${ROBOTWIN_SERVER_STORAGE_ROOT}/RoboTwin_geo"
    ROBOTWIN_RAW_DATA_ROOT=${ROBOTWIN_RAW_DATA_ROOT:-${_robotwin_storage_repo}/data}
    ROBOTWIN_DP3_DATA_ROOT=${ROBOTWIN_DP3_DATA_ROOT:-${_robotwin_storage_repo}/policy/DP3/data}
else
    ROBOTWIN_STORAGE_MODE=local
    ROBOTWIN_RAW_DATA_ROOT=${ROBOTWIN_RAW_DATA_ROOT:-${ROBOTWIN_REPO_ROOT}/data}
    ROBOTWIN_DP3_DATA_ROOT=${ROBOTWIN_DP3_DATA_ROOT:-${ROBOTWIN_REPO_ROOT}/policy/DP3/data}
fi

export ROBOTWIN_REPO_ROOT
export ROBOTWIN_SERVER_STORAGE_ROOT
export ROBOTWIN_STORAGE_MODE
export ROBOTWIN_RAW_DATA_ROOT
export ROBOTWIN_DP3_DATA_ROOT

unset _robotwin_data_paths_dir
unset _robotwin_storage_repo
