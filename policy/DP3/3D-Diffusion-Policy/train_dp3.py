if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os, sys
import contextlib
import pdb
import hydra
import torch
import dill
from collections import OrderedDict
from omegaconf import OmegaConf
import pathlib

DP3_ROOT = str(pathlib.Path(__file__).parent.parent)

sys.path.append(DP3_ROOT)
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy'))
sys.path.append(os.path.join(DP3_ROOT, '3D-Diffusion-Policy', 'diffusion_policy_3d'))

from torch.utils.data import DataLoader
import copy

import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import sys

from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from scripts.safe_dp3_checkpoint import (
    load_weights_only_checkpoint,
    save_weights_only_checkpoint,
    validate_policy_checkpoint_metadata,
)
from scripts.semantic_fusion_warmstart import (
    freeze_vanilla_backbone_for_adapter,
    load_shared_vanilla_state,
    validate_vanilla_warmstart_metadata,
)

import pdb, random

OmegaConf.register_new_resolver("eval", eval, replace=True)


SUPPORTED_TRAINING_PRECISIONS = {"fp32", "bf16"}


def resolve_training_precision(training_cfg) -> str:
    value = training_cfg.get("precision", "fp32")
    precision = "fp32" if value is None else str(value).strip().lower()
    if precision not in SUPPORTED_TRAINING_PRECISIONS:
        raise ValueError(
            f"training.precision must be one of {sorted(SUPPORTED_TRAINING_PRECISIONS)}, "
            f"got {value!r}"
        )
    return precision


def configure_training_tf32(training_cfg) -> bool | None:
    """Apply an explicit TF32 override, or preserve PyTorch defaults when absent/null."""
    requested = training_cfg.get("allow_tf32", None)
    if requested is None:
        return None
    if not isinstance(requested, bool):
        raise ValueError(f"training.allow_tf32 must be true, false, or null; got {requested!r}")
    torch.backends.cuda.matmul.allow_tf32 = requested
    torch.backends.cudnn.allow_tf32 = requested
    return requested


def training_autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unsupported training precision: {precision!r}")
    if device.type != "cuda":
        raise ValueError("training.precision=bf16 currently requires a CUDA device")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class TrainDP3Workspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        self.warmstart_provenance = None
        warmstart_value = str(
            cfg.training.get("warmstart_safe_checkpoint", "") or ""
        ).strip()
        freeze_vanilla_backbone = bool(
            cfg.training.get("freeze_vanilla_backbone", False)
        )
        warmstart_use_ema = bool(
            cfg.training.get("warmstart_use_ema_weights", True)
        )
        if freeze_vanilla_backbone and not warmstart_value:
            raise ValueError(
                "training.freeze_vanilla_backbone=true requires "
                "training.warmstart_safe_checkpoint"
            )
        if warmstart_value:
            if str(cfg.policy.get("pointcloud_fusion_mode", "concat")) == "concat":
                raise ValueError(
                    "Vanilla warm-start is only valid for a semantic fusion policy"
                )
            warmstart_path = pathlib.Path(warmstart_value).expanduser().resolve()
            if not warmstart_path.is_file():
                raise FileNotFoundError(
                    f"Warm-start safe checkpoint does not exist: {warmstart_path}"
                )
            warmstart_payload = load_weights_only_checkpoint(
                warmstart_path,
                mmap=True,
            )
            raw_task_name = str(
                cfg.get("raw_task_name", "") or ""
            ).strip()
            if raw_task_name.lower() in {"", "none", "null"}:
                raw_task_name = str(cfg.task_name)
            source_metadata = validate_vanilla_warmstart_metadata(
                warmstart_payload["metadata"],
                target_shape_meta=OmegaConf.to_container(
                    cfg.task.shape_meta,
                    resolve=True,
                ),
                expected_raw_task_name=raw_task_name,
                require_ema=warmstart_use_ema,
            )
            if warmstart_use_ema:
                source_model_state = warmstart_payload.get("ema_model")
                if source_model_state is None:
                    raise ValueError(
                        "Warm-start checkpoint does not contain EMA weights"
                    )
            else:
                source_model_state = warmstart_payload["model"]
            model_load_report = load_shared_vanilla_state(
                self.model,
                source_model_state,
            )
            ema_load_report = None
            if self.ema_model is not None:
                if warmstart_use_ema:
                    source_ema_state = source_model_state
                else:
                    source_ema_state = (
                        warmstart_payload.get("ema_model")
                        or warmstart_payload["model"]
                    )
                ema_load_report = load_shared_vanilla_state(
                    self.ema_model,
                    source_ema_state,
                )
            freeze_report = None
            if freeze_vanilla_backbone:
                freeze_report = freeze_vanilla_backbone_for_adapter(self.model)
            self.warmstart_provenance = {
                "source_path": str(warmstart_path),
                "use_ema_weights": warmstart_use_ema,
                "freeze_vanilla_backbone": freeze_vanilla_backbone,
                "source_metadata": source_metadata,
                "model_load": model_load_report,
                "ema_load": ema_load_report,
                "freeze": freeze_report,
            }
            print(f"[fusion-warmstart] {self.warmstart_provenance}")
            del warmstart_payload

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("DP3 optimizer has no trainable parameters")
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=trainable_parameters,
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        precision = resolve_training_precision(cfg.training)
        allow_tf32 = configure_training_tf32(cfg.training)
        if precision == "bf16":
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("training.precision=bf16 requires an available CUDA device")
            if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support bfloat16 training")
        print(
            f"[training-precision] precision={precision} "
            f"allow_tf32={allow_tf32 if allow_tf32 is not None else 'torch-default'}"
        )

        WANDB = False

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_ROLLOUT = False
        RUN_VALIDATION = True  # reduce time cost

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None

        cfg.logging.name = str(cfg.task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({
                "output_dir": self.output_dir,
            })

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        checkpoint_num = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    with training_autocast_context(device, precision):
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            with training_autocast_context(device, precision):
                                loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            print(f"epoch {self.epoch}, eval loss: ", float(loss.cpu()))
                            if (cfg.training.max_val_steps
                                    is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log["val_loss"] = val_loss

            # checkpoint
            if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:

                if not cfg.policy.use_pc_color:
                    if not os.path.exists(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}"):
                        os.makedirs(f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}")
                    save_path = f"checkpoints/{self.cfg.task.name}_{cfg.training.seed}/{self.epoch + 1}.ckpt"
                else:
                    if not os.path.exists(f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}"):
                        os.makedirs(f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}")
                    save_path = f"checkpoints/{self.cfg.task.name}_w_rgb_{cfg.training.seed}/{self.epoch + 1}.ckpt"

                self.save_checkpoint(save_path)
            weights_only_path = str(cfg.checkpoint.get("weights_only_path", "") or "").strip()
            if (
                ((self.epoch + 1) % cfg.training.checkpoint_every) == 0
                and bool(cfg.checkpoint.get("save_weights_only", False))
            ):
                if not weights_only_path:
                    raise ValueError("checkpoint.weights_only_path is required when save_weights_only=true")
                resolved_weights_path = weights_only_path.format(epoch=self.epoch + 1)
                metadata = {
                    "epoch": int(self.epoch + 1),
                    "global_step": int(self.global_step),
                    "seed": int(cfg.training.seed),
                    "task_name": str(cfg.task.name),
                    "zarr_path": str(cfg.task.dataset.zarr_path),
                    "precision": precision,
                    "allow_tf32": allow_tf32,
                    "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
                    "use_ema": bool(cfg.training.use_ema),
                    "pointcloud_fusion_mode": str(cfg.policy.get("pointcloud_fusion_mode", "concat")),
                    "semantic_gate_trainable": bool(cfg.policy.get("semantic_gate_trainable", True)),
                    "semantic_gate_scale": float(cfg.policy.get("semantic_gate_scale", 1.0)),
                    "semantic_attention_heads": int(cfg.policy.get("semantic_attention_heads", 4)),
                    "part_pool_temperature": float(cfg.policy.get("part_pool_temperature", 1.0)),
                    "warmstart": copy.deepcopy(self.warmstart_provenance),
                }
                saved_weights_path = save_weights_only_checkpoint(
                    resolved_weights_path,
                    model=self.model,
                    ema_model=self.ema_model,
                    metadata=metadata,
                )
                print(f"saved weights-only checkpoint in {saved_weights_path}")

            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def get_policy_and_runner(self, cfg, usr_args):
        # load the latest checkpoint

        cfg = copy.deepcopy(self.cfg)

        n_obs_steps = cfg['n_obs_steps']
        n_action_steps = cfg['n_action_steps']

        env_runner = RobotRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

        safe_checkpoint_value = str(usr_args.get("safe_checkpoint_path", "") or "").strip()
        if safe_checkpoint_value:
            ckpt_file = pathlib.Path(safe_checkpoint_value).expanduser().resolve()
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"safe DP3 checkpoint does not exist: {ckpt_file}")
            payload = load_weights_only_checkpoint(ckpt_file, mmap=True)
            validate_policy_checkpoint_metadata(
                payload,
                expected_task_name=str(cfg.task.name),
                expected_use_ema=bool(cfg.training.use_ema),
                expected_shape_meta=OmegaConf.to_container(
                    cfg.task.shape_meta,
                    resolve=True,
                ),
            )
            model_state = self._upgrade_legacy_model_state_dict(payload["model"])
            self.model.load_state_dict(model_state, strict=True)
            if cfg.training.use_ema:
                if self.ema_model is None:
                    raise ValueError(
                        "Safe DP3 checkpoint requests EMA but workspace has no ema_model"
                    )
                ema_state = self._upgrade_legacy_model_state_dict(payload["ema_model"])
                self.ema_model.load_state_dict(ema_state, strict=True)
            cprint(f"Loaded safe weights-only checkpoint {ckpt_file}", "magenta")
            del payload
        else:
            if not cfg.policy.use_pc_color:
                ckpt_file = pathlib.Path(
                    os.path.join(
                        DP3_ROOT,
                        f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
                    ))
            else:
                ckpt_file = pathlib.Path(
                    os.path.join(
                        DP3_ROOT,
                        f"./checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}_w_rgb_{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
                    ))
            assert ckpt_file.is_file(), f"ckpt file doesn't exist, {ckpt_file}"

            if ckpt_file.is_file():
                cprint(f"Resuming from checkpoint {ckpt_file}", "magenta")
                self.load_checkpoint(path=ckpt_file)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        print("saved in ", path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir", )

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                score = float(ckpt.split("test_mean_score=")[1].split(".ckpt")[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                if key in {"model", "ema_model"}:
                    value = self._upgrade_legacy_model_state_dict(value)
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    @staticmethod
    def _upgrade_legacy_model_state_dict(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict

        legacy_prefix = "obs_encoder.extractor."
        new_prefix = "obs_encoder.extractors.point_cloud."
        has_legacy = any(key.startswith(legacy_prefix) for key in state_dict.keys())
        has_new = any(key.startswith(new_prefix) for key in state_dict.keys())
        if not has_legacy or has_new:
            return state_dict

        upgraded = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith(legacy_prefix):
                key = key.replace(legacy_prefix, new_prefix, 1)
            upgraded[key] = value
        return upgraded

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")),
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
