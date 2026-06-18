from dataclasses import dataclass, field
from sc_evae.config.dataset import DatasetConfig
from sc_evae.config.models import sc_evaeConfig, ModelConfig


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "sc_evae"
    entity: str | None = None
    # Upload checkpoint as W&B artifact every N steps. Final checkpoint is
    # always uploaded regardless of this value. Must be a multiple of save_freq.
    # ``None`` resolves to ``2 * save_freq`` in TrainConfig.__post_init__.
    artifact_freq: int | None = None


@dataclass
class TrainConfig:
    work_dir: str = "outputs/experiments-nips"
    run_name: str = "run"
    prefix_run_name_with_datetime: bool = True
    seed: int = 42

    # Optimization
    n_iters: int = 100_000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    warmup_steps: int = 2_000
    # LR schedule after warmup: "constant" | "cosine" | "linear"
    lr_schedule: str = "constant"

    # Logging / checkpointing (in steps)
    log_freq: int = 100
    eval_freq: int = 5_000
    save_freq: int = 5_000
    # Cap validation at N batches per eval pass (None = full val set).
    max_val_iters: int | None = None

    # Accelerate mixed precision: "no" | "fp16" | "bf16"
    mixed_precision: str = "bf16"

    # Smoke-test mode: cap training at 100 iters and validation at 10 batches,
    # and force a final validation pass before the script exits so model.predict
    # gets exercised end-to-end.
    dry_run: bool = False

    # Per-section wall-clock profiling. When True, ``sc_evae.training.profiler``
    # records timings for ``profile_n_steps`` measured steps (after
    # ``profile_warmup_steps`` warmup steps that exclude kernel compile / cudnn
    # benchmark spikes), then dumps ``profile.json`` to the experiment dir,
    # prints a table summary, and disables itself. Training continues after
    # finalisation with zero overhead.
    profile: bool = False
    profile_n_steps: int = 500
    profile_warmup_steps: int = 50

    # Nested configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=sc_evaeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        # set is_count_discrete from dataset config — binning produces integer
        # bin IDs (discrete) regardless of whether log1p was applied beforehand.
        if self.model.is_count_discrete is None:
            self.model.is_count_discrete = (
                self.dataset.use_binning != "none" or not self.dataset.apply_log1p
            )
        if self.dry_run:
            self.n_iters = 100
        if self.wandb.artifact_freq is None:
            self.wandb.artifact_freq = 2 * self.save_freq
        elif (
            self.wandb.enabled
            and self.wandb.artifact_freq > 0
            and self.wandb.artifact_freq % self.save_freq != 0
        ):
            raise ValueError(
                f"wandb.artifact_freq ({self.wandb.artifact_freq}) must be a "
                f"multiple of save_freq ({self.save_freq})."
            )
