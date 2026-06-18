"""
scripts/train.py
----------------
VAE training script using Accelerate + Draccus.

Single GPU:
    uv run accelerate launch scripts/train.py --config_path config/fsq_vae.yaml

Multi-GPU:
    uv run accelerate launch --num_processes 4 scripts/train.py --config_path config/fsq_vae.yaml

CLI flags override YAML values, which override dataclass defaults.
"""

import dataclasses
import json
import os


import draccus
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.optim import AdamW
from tqdm.auto import tqdm

from sc_evae.config.training import TrainConfig
from sc_evae.models.factory import make_model
from sc_evae.training.data_loader import (
    PerturbationDataset,
    make_dataloader_from_dataset,
)
from sc_evae.training.train_utils import (
    build_scheduler,
    cycle_loader,
    latest_checkpoint,
    log_checkpoint_artifact,
    prefix_metrics,
    reduce_metric_dict,
    resolve_config,
    resolve_deg_table,
    resolve_train_val_split,
    resolve_run_name,
    run_validation,
    save_checkpoint,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: TrainConfig) -> None:
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    set_seed(cfg.seed)
    base_run_name = cfg.run_name

    cfg.run_name = resolve_run_name(
        run_name=cfg.run_name,
        work_dir=cfg.work_dir,
        prefix_run_name_with_datetime=cfg.prefix_run_name_with_datetime,
        accelerator=accelerator,
    )

    # ---- directories -------------------------------------------------------
    experiment_dir = os.path.join(cfg.work_dir, cfg.run_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    log_dir = os.path.join(experiment_dir, "logs")
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ---- logging -----------------------------------------------------------
    setup_logging(log_dir, is_main_process=accelerator.is_main_process)
    logger = get_logger(__name__)

    def mlog(metrics: dict, step: int) -> None:
        if accelerator.is_main_process and cfg.wandb.enabled:
            wandb.log(metrics, step=step)

    logger.info(f"Config:\n{cfg}", main_process_only=True)
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=True)

    # ---- data + config resolution ------------------------------------------
    if not cfg.dataset.h5ad_path and not cfg.dataset.h5ad_paths:
        raise ValueError("cfg.dataset.h5ad_path or cfg.dataset.h5ad_paths must be set")
    _paths = cfg.dataset.h5ad_paths or [cfg.dataset.h5ad_path]
    logger.info(f"Loading dataset from {_paths}", main_process_only=True)
    ds = PerturbationDataset(config=cfg.dataset, logger=logger)

    # Compute stats before resolve_config so ds.count_max is populated (lazy mode
    # sets it as a side effect of the full scan done here). Use a smaller sample
    # when profiling so startup doesn't dominate the run.
    logger.info("Computing dataset statistics …", main_process_only=True)
    n_sample_cells = 1_000 if cfg.profile else 20_000
    data_stats = ds.compute_data_stats(n_sample_cells=n_sample_cells)

    cfg = resolve_config(cfg, ds)
    logger.info(
        f"Resolved: num_genes={cfg.model.num_genes}, "
        f"pert_emb_dim={getattr(cfg.model, 'pert_emb_dim', None)}, "
        f"num_cell_types={getattr(cfg.model, 'num_cell_types', None)}, "
        f"max_count={cfg.model.max_count}",
        main_process_only=True,
    )

    if accelerator.is_main_process:
        config_path = os.path.join(experiment_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            draccus.dump(cfg, f)
        logger.info(f"Saved run config to {config_path}", main_process_only=True)
        stats_path = os.path.join(experiment_dir, "dataset_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(data_stats), f, indent=2)
        logger.info(f"Saved dataset stats to {stats_path}", main_process_only=True)

    if accelerator.is_main_process and cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=base_run_name,
            config=draccus.encode(cfg),
            dir=experiment_dir,
        )

    train_ds, val_ds = resolve_train_val_split(ds, cfg.dataset, cfg.seed)

    # ---- DEG table ---------------------------------------------------------
    deg_table = resolve_deg_table(
        ds,
        cfg.dataset,
        experiment_dir,
        logger=logger,
        is_main_process=accelerator.is_main_process,
    )

    # ---- model -------------------------------------------------------------
    model = make_model(
        cfg.model,
        log_lib_prior_mu=data_stats.log_lib_mean,
        log_lib_prior_std=data_stats.log_lib_std,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}", main_process_only=True)
    logger.info(
        f"Model trainable parameters: {n_trainable_params:,}", main_process_only=True
    )

    # ---- optimizer + scheduler ---------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    # Each process gets its own subset via DistributedSampler (handled by accelerate).
    train_dl = make_dataloader_from_dataset(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
    )
    val_dl = make_dataloader_from_dataset(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    # ---- accelerate prepare ------------------------------------------------
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    # ---- checkpoint restore ------------------------------------------------
    initial_step = 0
    resume_path = latest_checkpoint(checkpoint_dir)
    if resume_path is not None:
        logger.info(f"Resuming from checkpoint: {resume_path}", main_process_only=True)
        accelerator.load_state(resume_path)
        # Step is encoded in the directory name: checkpoints/step_NNNNN
        try:
            initial_step = int(os.path.basename(resume_path).split("_")[1])
        except (ValueError, IndexError):
            pass
        else:
            # Checkpoint at step N means step N is already completed.
            initial_step += 1

    # ---- training loop -----------------------------------------------------
    logger.info(f"Starting training at step {initial_step}", main_process_only=True)
    steps_per_epoch = max(1, len(train_dl))
    train_iter = cycle_loader(train_dl)
    train_pbar = None
    if accelerator.is_main_process:
        initial_remaining = max(0, cfg.n_iters - initial_step + 1)
        train_pbar = tqdm(
            total=min(cfg.eval_freq, initial_remaining), desc="train", leave=True
        )

    # Per-section profiler. No-op when cfg.profile is False (zero overhead).
    from sc_evae.training.profiler import Profiler

    profiler = Profiler(
        enabled=cfg.profile,
        n_steps=cfg.profile_n_steps,
        warmup_steps=cfg.profile_warmup_steps,
        device=accelerator.device,
        batch_size=cfg.dataset.batch_size,
    )
    if cfg.profile:
        logger.info(
            f"Profiling enabled: {cfg.profile_warmup_steps} warmup + "
            f"{cfg.profile_n_steps} measured steps, then summary to "
            f"{os.path.join(experiment_dir, 'profile.json')}",
            main_process_only=True,
        )

    for step in range(initial_step, cfg.n_iters + 1):
        model.train()
        profiler.begin_step()

        with profiler.section("data_fetch"):
            batch = next(train_iter)

        with profiler.section("forward"):
            with accelerator.autocast():
                loss, metrics = model(batch)

        with profiler.section("backward"):
            accelerator.backward(loss)

        with profiler.section("optimizer"):
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip if cfg.grad_clip > 0 else float("inf"),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        lr_now = scheduler.get_last_lr()[0]
        epoch_now = step / steps_per_epoch
        with profiler.section("logging"):
            reduced_loss = reduce_metric_dict(accelerator, {"loss": loss})["loss"]
        if train_pbar is not None:
            train_pbar.update(1)
            train_pbar.set_postfix(
                step=step,
                epoch=f"{epoch_now:.2f}",
                loss=f"{reduced_loss:.4f}",
                lr=f"{lr_now:.2e}",
            )

        profiler.end_step()
        if profiler.should_finalize():
            profiler.finalize(out_dir=experiment_dir, logger=logger)

        # ---- logging -------------------------------------------------------
        if step % cfg.log_freq == 0:
            reduced = reduce_metric_dict(accelerator, {"loss": loss, **metrics})
            train_log = prefix_metrics(reduced, "train")
            train_log["train/lr"] = lr_now
            train_log["train/epoch"] = epoch_now
            train_log["train/grad_norm"] = grad_norm.item()
            mlog(train_log, step=step)

        # ---- eval ----------------------------------------------------------
        if step > 0 and step % cfg.eval_freq == 0:
            model.eval()
            val_metrics_mean = run_validation(
                model,
                val_dl,
                accelerator,
                deg_table,
                ds.unique_perts,
                cfg.dataset,
                max_batches=10 if cfg.dry_run else cfg.max_val_iters,
            )
            logger.info(
                f"step={step:07d}  val_loss={val_metrics_mean['loss']:.4f}",
                main_process_only=True,
            )
            mlog(prefix_metrics(val_metrics_mean, "val"), step=step)
            if train_pbar is not None and step < cfg.n_iters:
                train_pbar.close()
                remaining = max(0, cfg.n_iters - step)
                train_pbar = tqdm(
                    total=min(cfg.eval_freq, remaining), desc="train", leave=True
                )

        # ---- checkpoint ----------------------------------------------------
        if step > 0 and step % cfg.save_freq == 0:
            ckpt_name = f"step_{step:07d}"
            ckpt_path = save_checkpoint(accelerator, checkpoint_dir, ckpt_name)
            logger.info(f"Saved checkpoint: {ckpt_path}", main_process_only=True)
            if (
                accelerator.is_main_process
                and cfg.wandb.enabled
                and cfg.wandb.artifact_freq > 0
                and step % cfg.wandb.artifact_freq == 0
            ):
                log_checkpoint_artifact(ckpt_path, base_run_name, step)
                logger.info(
                    f"Logged W&B artifact for step {step}", main_process_only=True
                )

    # ---- profiler fallback: dump partial summary if the loop ended before
    # the configured profile window completed (e.g. dry_run capped n_iters).
    if profiler.enabled:
        profiler.finalize(out_dir=experiment_dir, logger=logger)

    # ---- final checkpoint --------------------------------------------------
    ckpt_name = f"step_{cfg.n_iters:07d}_final"
    ckpt_path = save_checkpoint(accelerator, checkpoint_dir, ckpt_name)
    logger.info(f"Saved final checkpoint: {ckpt_path}", main_process_only=True)
    if accelerator.is_main_process and cfg.wandb.enabled:
        log_checkpoint_artifact(
            ckpt_path, base_run_name, cfg.n_iters, extra_aliases=["final"]
        )
        logger.info("Logged final W&B artifact", main_process_only=True)
    if train_pbar is not None:
        train_pbar.close()

    # ---- dry-run final eval (exercises model.predict) ----------------------
    if cfg.dry_run:
        model.eval()
        val_metrics_mean = run_validation(
            model,
            val_dl,
            accelerator,
            deg_table,
            ds.unique_perts,
            cfg.dataset,
            max_batches=10,
        )
        logger.info(
            f"[dry_run] final val metrics: "
            + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics_mean.items()),
            main_process_only=True,
        )
        mlog(prefix_metrics(val_metrics_mean, "val"), step=cfg.n_iters)

    if accelerator.is_main_process and cfg.wandb.enabled:
        wandb.finish()

    logger.info("Training complete.", main_process_only=True)


if __name__ == "__main__":
    main()
