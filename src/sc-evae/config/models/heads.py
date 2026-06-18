"""Output- and input-head configs, plus head/data-compatibility helpers."""

from dataclasses import dataclass

from draccus import ChoiceRegistry

# ---------------------------------------------------------------------------
# Output head configs
# ---------------------------------------------------------------------------
# Each subclass carries only the parameters relevant to that head type.
# No stale fields from other head types leak in. Dispatch in model code with
# isinstance(config, CrossEntropyHeadConfig) — never string comparison.


class OutputHeadConfig(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "mse"


# --- Continuous-data heads (use with is_count_discrete=False) ---


@OutputHeadConfig.register_subclass("mse")
@dataclass
class MSEHeadConfig(OutputHeadConfig):
    """Direct MSE regression. No additional parameters."""


@OutputHeadConfig.register_subclass("hurdle")
@dataclass
class HurdleHeadConfig(OutputHeadConfig):
    """Binary zero/nonzero + MSE on positive entries.

    ``weight_for_bce_loss`` multiplies the BCE term.  Single-cell data is
    ~90% zeros, so the BCE head is an easy signal that can drown out the
    magnitude regression — tune this down (e.g. 0.1–0.3) if the hurdle
    fits zeros well but predicts flat magnitudes on nonzero entries.
    """

    weight_for_bce_loss: float = 0.1


# --- Discrete-data heads (use with is_count_discrete=True) ---


@OutputHeadConfig.register_subclass("nb")
@dataclass
class NBHeadConfig(OutputHeadConfig):
    """Negative Binomial (mu, theta) parametric head.

    ``temperature`` softens the rho softmax: ``rho = softmax(logits / T)``.
    Default 1.0 (no change).  Larger T (e.g. 1.5–2.0) softens rho — useful
    if training shows winner-take-all dynamics where some genes get
    rho ≈ 0 and produce gradient spikes through the NB likelihood.

    ``min_theta`` floors the per-gene dispersion to prevent collapse toward
    Poisson (theta → 0 makes log(mu / (theta + mu)) numerically explosive).
    Applied after ``exp(theta_embedding)``; default 1e-2 is well below
    typical dispersion (~1–10) so the floor only kicks in if a gene's
    theta is collapsing.
    """

    temperature: float = 1.0
    min_theta: float = 1e-2


@OutputHeadConfig.register_subclass("cross_entropy")
@dataclass
class CrossEntropyHeadConfig(OutputHeadConfig):
    """Non-parametric categorical over [0, max_count]."""

    max_count: int = 4096


# ---------------------------------------------------------------------------
# Input head configs
# ---------------------------------------------------------------------------
# Projects a scalar expression count per gene to ``embed_dim`` features.  Each
# subclass carries only its own parameters.  Dispatch in model code with
# isinstance(config, EmbeddingInputHeadConfig) — never string comparison.


class InputHeadConfig(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "sinusoidal"


@InputHeadConfig.register_subclass("mlp")
@dataclass
class MLPInputHeadConfig(InputHeadConfig):
    """2-layer MLP on the scalar count.

    Handles both continuous log1p inputs (``is_count_discrete=False``, no
    internal transform) and raw integer counts (``is_count_discrete=True``,
    log1p applied inside ``GeneCountEmbedding``).  ``is_count_discrete`` is
    auto-resolved from the enclosing ``ModelConfig`` at
    ``set_attributes_from_dataset`` time — do not set it in YAML.
    """

    is_count_discrete: bool | None = None


@InputHeadConfig.register_subclass("sinusoidal")
@dataclass
class SinusoidalInputHeadConfig(InputHeadConfig):
    """Sinusoidal features + learned linear projection.  Works for either
    continuous or discrete inputs.  ``scale`` multiplies the count before
    encoding so it lands in the useful frequency range.

      * log1p-normalised counts (~0-8)   → scale ≈ 1000
      * raw integer counts     (~0-3000) → scale = 1
    """

    scale: float = 1.0


@InputHeadConfig.register_subclass("embedding")
@dataclass
class EmbeddingInputHeadConfig(InputHeadConfig):
    """Learned embedding table over integer counts, clamped at ``max_count``.
    Discrete inputs only.  ``max_count`` is auto-resolved from the dataset."""

    max_count: int = 4096


_DISCRETE_HEAD_TYPES = (NBHeadConfig, CrossEntropyHeadConfig)
_CONTINUOUS_HEAD_TYPES = (MSEHeadConfig, HurdleHeadConfig)


def _validate_input_head(
    input_head: InputHeadConfig,
    is_count_discrete: bool | None,
    config_name: str,
) -> None:
    """
    Check compatibility between ``input_head`` type and ``is_count_discrete``.
    Safe to call with ``is_count_discrete=None`` — it becomes a no-op, letting
    validation run again after ``resolve_config_dependencies`` resolves the flag.
    """
    if is_count_discrete is None:
        return
    # MLP + is_count_discrete=True is fine: GeneCountEmbedding applies log1p
    # inside the head so the MLP always sees a smooth continuous scalar.
    if isinstance(input_head, EmbeddingInputHeadConfig) and not is_count_discrete:
        raise ValueError(
            f"{config_name}.input_head=EmbeddingInputHeadConfig requires discrete "
            f"counts (is_count_discrete=True).  Use MLPInputHeadConfig or "
            f"SinusoidalInputHeadConfig for continuous log1p values."
        )


def _propagate_count_max(
    count_max: int | None,
    *targets,
) -> None:
    """Write ``count_max`` into every target's ``max_count`` attribute (if present)."""
    if count_max is None:
        return
    for t in targets:
        if t is not None and hasattr(t, "max_count"):
            t.max_count = count_max


def _propagate_is_count_discrete(
    is_count_discrete: bool | None,
    *targets,
) -> None:
    """Write ``is_count_discrete`` into every target that carries the field.

    Only ``MLPInputHeadConfig`` currently does; other input-head variants
    don't need the flag (sinusoidal tunes via ``scale``; embedding takes
    integer indices by construction).
    """
    if is_count_discrete is None:
        return
    for t in targets:
        if t is not None and hasattr(t, "is_count_discrete"):
            t.is_count_discrete = is_count_discrete
