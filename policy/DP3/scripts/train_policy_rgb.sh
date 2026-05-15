DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
setting=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${7}
dataloader_num_workers=${8:-4}
val_dataloader_num_workers=${9:-2}
pin_memory=${10:-true}
val_pin_memory=${11:-false}
max_val_steps=${12:-2}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=${config_name}.yaml \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            setting=${setting} \
                            dataloader.num_workers=${dataloader_num_workers} \
                            dataloader.pin_memory=${pin_memory} \
                            val_dataloader.num_workers=${val_dataloader_num_workers} \
                            val_dataloader.pin_memory=${val_pin_memory} \
                            training.max_val_steps=${max_val_steps} \
                            policy.use_pc_color=True 
