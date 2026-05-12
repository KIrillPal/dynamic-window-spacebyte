from __future__ import annotations

import contextlib
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .config import RouterTrainConfig
from .dataset import CombinedSpaceByteRouterDataset
from .model import RouterModelProtocol


class RouterTrainer:
    def __init__(
        self,
        cfg: RouterTrainConfig,
        model: RouterModelProtocol,
        train_dataset: CombinedSpaceByteRouterDataset,
        val_dataset: CombinedSpaceByteRouterDataset | None = None,
    ) -> None:
        self.cfg = cfg
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.is_main = self.rank == 0
        self.device = self._resolve_device()
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.global_step = 0
        self.best_metric = math.inf if cfg.save_best_mode == "min" else -math.inf
        self.mlflow_module = None

        if cfg.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        if cfg.freeze_encoder:
            self.model.freeze_encoder(unfreeze_last_n_layers=cfg.unfreeze_last_n_layers)
        if cfg.train_classifier_only:
            self.model.freeze_regressor_head()
        elif cfg.train_regressor_only:
            self.model.freeze_classifier_head()

        if self.world_size > 1:
            # Some batches have has_regression all False → regression loss is detached from regressor
            # (see compute_loss); frozen encoder may also skip grads. DDP needs unused-parameter detection.
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        self.optimizer = self._build_optimizer()
        self.train_loader = self._build_loader(train_dataset, cfg.train_batch_size, train=True)
        self.val_loader = (
            self._build_loader(val_dataset, cfg.eval_batch_size, train=False)
            if val_dataset is not None
            else None
        )
        updates_per_epoch = math.ceil(len(self.train_loader) / cfg.gradient_accumulation_steps)
        total_steps = max(1, updates_per_epoch * cfg.epochs)
        if cfg.warmup_steps > 0:
            warmup_steps = cfg.warmup_steps
        elif cfg.warmup_ratio > 0:
            warmup_steps = int(total_steps * cfg.warmup_ratio)
        else:
            warmup_steps = 0
        self.scheduler = build_scheduler(cfg, self.optimizer, warmup_steps, total_steps)

        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    def train(self) -> None:
        cfg = self.cfg
        output_dir = Path(cfg.output_dir)
        if self.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg.save(output_dir / "config.resolved.yaml")
            self._start_mlflow()

        try:
            for epoch in range(cfg.epochs):
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)
                metrics = self._train_epoch(epoch)
                if self.is_main:
                    self._log_metrics(metrics, self.global_step)

                if self.val_loader is not None:
                    val_metrics = self.evaluate(prefix="val")
                    if self.is_main:
                        self._log_metrics(val_metrics, self.global_step)
                        print(format_metrics(f"epoch {epoch + 1} val", val_metrics), flush=True)
                        self._maybe_save_best(val_metrics)

            if self.is_main:
                self._save_checkpoint(Path(cfg.output_dir) / "checkpoint_last.pt")
        finally:
            if self.mlflow_module is not None:
                self.mlflow_module.end_run()
            if self.world_size > 1:
                dist.barrier()
                dist.destroy_process_group()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        cfg = self.cfg
        acc = MetricAccumulator(self.device)
        pbar = tqdm(self.train_loader, disable=not self.is_main, desc=f"epoch {epoch + 1}")
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            batch = move_batch(batch, self.device)
            with autocast_context(self.device, cfg.precision):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                loss, parts = compute_loss(cfg, outputs, batch)
                loss_to_backward = loss / cfg.gradient_accumulation_steps

            if cfg.precision == "fp16" and self.device.type == "cuda":
                scaler = getattr(self, "_scaler", None)
                if scaler is None:
                    self._scaler = torch.cuda.amp.GradScaler()
                    scaler = self._scaler
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            update_metrics(acc, outputs, batch, parts)

            do_update = step % cfg.gradient_accumulation_steps == 0 or step == len(self.train_loader)
            if do_update:
                if cfg.precision == "fp16" and self.device.type == "cuda":
                    scaler = self._scaler
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._unwrap().parameters(), cfg.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self._unwrap().parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # acc.compute() uses all_reduce — every rank must call it, not only rank 0.
                if cfg.log_every_steps > 0 and self.global_step % cfg.log_every_steps == 0:
                    metrics = acc.compute("train")
                    if self.is_main:
                        self._log_metrics(metrics, self.global_step)
                        self._log_metrics(
                            {
                                "train/classification_loss_batch": float(parts["cls_loss"].item()),
                                "train/regression_loss_batch": float(parts["reg_loss"].item()),
                            },
                            self.global_step,
                        )
                        pbar.set_postfix(
                            {k: f"{v:.4f}" for k, v in metrics.items() if k in {"train/loss", "train/acc"}}
                        )

                if (
                    self.val_loader is not None
                    and cfg.eval_every_steps > 0
                    and self.global_step % cfg.eval_every_steps == 0
                ):
                    val_metrics = self.evaluate(prefix="val")
                    if self.is_main:
                        self._log_metrics(val_metrics, self.global_step)
                        self._maybe_save_best(val_metrics)
                    self.model.train()

                if cfg.save_every_steps > 0 and self.global_step % cfg.save_every_steps == 0 and self.is_main:
                    self._save_checkpoint(Path(cfg.output_dir) / f"checkpoint_step_{self.global_step}.pt")

        return acc.compute("train")

    @torch.inference_mode()
    def evaluate(self, prefix: str = "val") -> dict[str, float]:
        assert self.val_loader is not None
        self.model.eval()
        acc = MetricAccumulator(self.device)
        for batch in tqdm(self.val_loader, disable=not self.is_main, desc=prefix):
            batch = move_batch(batch, self.device)
            with autocast_context(self.device, self.cfg.precision):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                _loss, parts = compute_loss(self.cfg, outputs, batch)
            update_metrics(acc, outputs, batch, parts)
        return acc.compute(prefix)

    def _build_loader(
        self,
        dataset: CombinedSpaceByteRouterDataset,
        batch_size: int,
        *,
        train: bool,
    ) -> DataLoader:
        sampler = DistributedSampler(dataset, shuffle=train) if self.world_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _build_optimizer(self) -> AdamW:
        cfg = self.cfg
        model = self._unwrap()
        encoder_lr = float(cfg.encoder_learning_rate if cfg.encoder_learning_rate is not None else cfg.learning_rate)
        head_lr = float(cfg.head_learning_rate if cfg.head_learning_rate is not None else cfg.learning_rate)
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

        groups: list[dict[str, Any]] = []
        for prefix, module, lr in [
            ("encoder", model.encoder, encoder_lr),
            ("classifier", model.classifier, head_lr),
            ("regressor", model.regressor, head_lr),
        ]:
            decay_params = []
            nodecay_params = []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{prefix}.{name}"
                if any(nd in full_name for nd in no_decay):
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
            if decay_params:
                groups.append({"params": decay_params, "lr": lr, "weight_decay": cfg.weight_decay})
            if nodecay_params:
                groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})

        return AdamW(groups, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps)

    def _resolve_device(self) -> torch.device:
        if self.cfg.device:
            return torch.device(self.cfg.device)
        if torch.cuda.is_available():
            if self.world_size > 1:
                torch.cuda.set_device(self.local_rank)
                return torch.device("cuda", self.local_rank)
            return torch.device("cuda")
        return torch.device("cpu")

    def _unwrap(self) -> RouterModelProtocol:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._unwrap().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.cfg.to_dict(),
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            },
            path,
        )
        if self.mlflow_module is not None:
            self.mlflow_module.log_artifact(str(path))

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._unwrap().load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))

    def _maybe_save_best(self, metrics: dict[str, float]) -> None:
        key = self.cfg.save_best_metric
        if key not in metrics:
            return
        value = metrics[key]
        better = value < self.best_metric if self.cfg.save_best_mode == "min" else value > self.best_metric
        if better:
            self.best_metric = value
            self._save_checkpoint(Path(self.cfg.output_dir) / "checkpoint_best.pt")

    def _start_mlflow(self) -> None:
        if not self.cfg.mlflow:
            return
        import mlflow

        self.mlflow_module = mlflow
        if self.cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow_experiment)
        mlflow.start_run(run_name=self.cfg.mlflow_run_name)
        mlflow.log_params(_flatten_params(self.cfg.to_dict()))

    def _log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self.mlflow_module is not None:
            self.mlflow_module.log_metrics(metrics, step=step)


def classification_cross_entropy_loss(
    logits: torch.Tensor,
    target_long: torch.Tensor,
    *,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    """CE in fp32 with autocast disabled — bf16/fp16 softmax + CE are often unstable for training."""
    ctx = torch.amp.autocast("cuda", enabled=False) if logits.is_cuda else contextlib.nullcontext()
    with ctx:
        return F.cross_entropy(logits.float(), target_long, weight=class_weight)


def compute_loss(
    cfg: RouterTrainConfig,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cls_target = batch["use_global_blocks"]
    ratio_target = batch["local_window_fraction"]
    regression_mask = batch.get("has_regression")
    if regression_mask is None:
        regression_mask = torch.ones_like(ratio_target)
    regression_mask = regression_mask.to(dtype=torch.bool)

    logits = outputs["global_logits"]
    cls_target_long = cls_target.long().reshape(-1).clamp(0, 1)
    ce_weight = None
    if cfg.pos_weight is not None:
        ce_weight = torch.tensor(
            [1.0, float(cfg.pos_weight)],
            device=logits.device,
            dtype=torch.float32,
        )
    cls_loss = classification_cross_entropy_loss(logits, cls_target_long, class_weight=ce_weight)

    if regression_mask.any():
        pred_ratio = outputs["window_ratio"][regression_mask]
        target_ratio = ratio_target[regression_mask]
    else:
        pred_ratio = outputs["window_ratio"][:0]
        target_ratio = ratio_target[:0]

    if cfg.regression_loss == "mse":
        reg_loss = (
            F.mse_loss(pred_ratio, target_ratio)
            if regression_mask.any()
            else outputs["window_ratio"].sum() * 0.0
        )
    elif cfg.regression_loss == "mae":
        reg_loss = (
            F.l1_loss(pred_ratio, target_ratio)
            if regression_mask.any()
            else outputs["window_ratio"].sum() * 0.0
        )
    elif cfg.regression_loss in {"huber", "smooth_l1"}:
        reg_loss = (
            F.huber_loss(pred_ratio, target_ratio, delta=cfg.huber_delta)
            if regression_mask.any()
            else outputs["window_ratio"].sum() * 0.0
        )
    else:
        raise ValueError(f"Unknown regression_loss={cfg.regression_loss!r}")

    if cfg.train_classifier_only:
        loss = cfg.classification_loss_weight * cls_loss
    elif cfg.train_regressor_only:
        loss = cfg.regression_loss_weight * reg_loss
    else:
        loss = cfg.classification_loss_weight * cls_loss + cfg.regression_loss_weight * reg_loss
    return loss, {
        "loss": loss.detach(),
        "cls_loss": cls_loss.detach(),
        "reg_loss": reg_loss.detach(),
        "reg_n": regression_mask.sum().detach().to(torch.float32),
    }


def update_metrics(
    acc: "MetricAccumulator",
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
) -> None:
    logits = outputs["global_logits"].detach()
    pred = logits.argmax(dim=-1).to(torch.float32)
    target = batch["use_global_blocks"].to(torch.float32)
    ratio = outputs["window_ratio"].detach()
    ratio_target = batch["local_window_fraction"].to(torch.float32)
    regression_mask = batch.get("has_regression")
    if regression_mask is None:
        regression_mask = torch.ones_like(ratio_target)
    regression_mask = regression_mask.to(dtype=torch.bool)
    n = target.numel()
    reg_n = regression_mask.sum()

    acc.add("loss_sum", losses["loss"] * n)
    acc.add("cls_loss_sum", losses["cls_loss"] * n)
    acc.add("reg_loss_sum", losses["reg_loss"] * reg_n)
    acc.add("n", torch.tensor(float(n), device=target.device))
    acc.add("reg_n", reg_n.to(torch.float32))
    acc.add("correct", (pred == target).sum())
    acc.add("tp", ((pred == 1) & (target == 1)).sum())
    acc.add("fp", ((pred == 1) & (target == 0)).sum())
    acc.add("fn", ((pred == 0) & (target == 1)).sum())
    if regression_mask.any():
        acc.add("mse_sum", ((ratio[regression_mask] - ratio_target[regression_mask]) ** 2).sum())
        acc.add("mae_sum", (ratio[regression_mask] - ratio_target[regression_mask]).abs().sum())
    else:
        acc.add("mse_sum", ratio.sum() * 0.0)
        acc.add("mae_sum", ratio.sum() * 0.0)


class MetricAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.values: dict[str, torch.Tensor] = {}

    def add(self, key: str, value: torch.Tensor) -> None:
        value = value.detach().to(self.device, dtype=torch.float32)
        if key not in self.values:
            self.values[key] = value.clone()
        else:
            self.values[key] += value

    def compute(self, prefix: str) -> dict[str, float]:
        values = {k: v.clone() for k, v in self.values.items()}
        if dist.is_available() and dist.is_initialized():
            for value in values.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        n = values.get("n", torch.tensor(0.0, device=self.device)).clamp_min(1.0)
        reg_n_raw = values.get("reg_n", torch.tensor(0.0, device=self.device))
        reg_n = reg_n_raw.clamp_min(1.0)
        tp = values.get("tp", torch.tensor(0.0, device=self.device))
        fp = values.get("fp", torch.tensor(0.0, device=self.device))
        fn = values.get("fn", torch.tensor(0.0, device=self.device))
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        metrics = {
            f"{prefix}/loss": values["loss_sum"] / n,
            f"{prefix}/classification_loss": values["cls_loss_sum"] / n,
            f"{prefix}/regression_loss": values["reg_loss_sum"] / reg_n,
            f"{prefix}/acc": values["correct"] / n,
            f"{prefix}/precision": precision,
            f"{prefix}/recall": recall,
            f"{prefix}/f1": f1,
            f"{prefix}/mse": values["mse_sum"] / reg_n,
            f"{prefix}/mae": values["mae_sum"] / reg_n,
            f"{prefix}/regression_samples": reg_n_raw,
        }
        return {k: float(v.item()) for k, v in metrics.items()}


def build_scheduler(
    cfg: RouterTrainConfig,
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if cfg.scheduler == "constant":
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if cfg.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        if cfg.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if cfg.scheduler == "cosine_with_restarts":
            cycles = max(cfg.num_cycles, 1e-8)
            return 0.5 * (1.0 + math.cos(math.pi * ((cycles * progress) % 1.0)))
        if cfg.scheduler == "polynomial":
            return (1.0 - progress) ** 2
        if cfg.scheduler == "constant_with_warmup":
            return 1.0
        raise ValueError(f"Unknown scheduler={cfg.scheduler!r}")

    return LambdaLR(optimizer, lr_lambda)


def format_metrics(title: str, metrics: dict[str, float]) -> str:
    ordered_suffixes = (
        "loss",
        "classification_loss",
        "regression_loss",
        "acc",
        "f1",
        "mse",
        "mae",
        "regression_samples",
        "precision",
        "recall",
    )
    parts = []
    for suffix in ordered_suffixes:
        for key, value in metrics.items():
            if key.endswith("/" + suffix):
                parts.append(f"{key}={value:.6g}")
                break
    return f"{title}: " + " ".join(parts)


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    return flat
