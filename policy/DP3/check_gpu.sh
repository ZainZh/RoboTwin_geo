#!/bin/bash

# 定义你的训练命令
TRAIN_COMMAND="bash train_semantic_pointwise_hybrid_eef_absolute6d_global.sh stirring_coffee demo_real_zed_sam2_objpc_global 80 0 0 /home/zheng/model/semantic/mug.pt /home/zheng/model/semantic/spoon.pt"

# 目标空闲内存 (MiB)
REQUIRED_FREE_MEMORY=21500

# 要监测的 GPU ID
GPU_ID=0

while true; do
    # 获取 GPU 1 的空闲内存信息
    # 使用 awk 提取 free MiB，并去掉 "MiB" 后缀
    FREE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")

    echo "GPU $GPU_ID 当前空闲内存: $FREE_MEMORY MiB"

    # 检查空闲内存是否大于等于所需值
    if (( FREE_MEMORY >= REQUIRED_FREE_MEMORY )); then
        echo "GPU $GPU_ID 空闲内存达到 $REQUIRED_FREE_MEMORY MiB，开始执行命令..."
        eval "$TRAIN_COMMAND"
        echo "命令执行完毕。"
        # 如果你只想运行一次，可以在这里添加 exit
        # exit 0
        break # 如果你只想运行一次，执行完就退出循环
    else
        echo "GPU $GPU_ID 空闲内存不足，等待 60 秒后再次检查..."
        sleep 60
    fi
done