"""Bottleneck (quantizer) configs: fsq (default), vq, lfq, noquant."""

import math
from dataclasses import dataclass

from draccus import ChoiceRegistry


def get_fsq_level(codebook_size: int) -> list[int]:
    power = int(math.log2(codebook_size))
    if codebook_size != 2**power:
        raise ValueError(
            f"codebook_size={codebook_size} is not a power of 2; "
            f"supported sizes: 16, 64, 256, 512, 1024, 2048, 4096"
        )
    if power == 4:
        return [5, 3]
    elif power == 6:
        return [8, 8]
    elif power == 8:
        return [8, 6, 5]
    elif power == 9:
        return [8, 8, 8]
    elif power == 10:
        return [8, 5, 5, 5]
    elif power == 11:
        return [8, 8, 6, 5]
    elif power == 12:
        return [7, 5, 5, 5, 5]
    raise ValueError(
        f"No FSQ level mapping for codebook_size={codebook_size} (power={power})"
    )


class QuantizerConfig(ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "fsq"


@QuantizerConfig.register_subclass("vq")
@dataclass
class VQConfig(QuantizerConfig):
    codebook_size: int = 512
    codebook_dim: int = 32
    commitment_weight: float = 1.0


@QuantizerConfig.register_subclass("fsq")
@dataclass
class FSQConfig(QuantizerConfig):
    codebook_size: int = 512
    fsq_level: list[int] | None = None

    def __post_init__(self):
        if self.fsq_level is None:
            self.fsq_level = get_fsq_level(self.codebook_size)
        self.codebook_size = math.prod(self.fsq_level)

    @property
    def codebook_dim(self) -> int:
        return len(self.fsq_level)


@QuantizerConfig.register_subclass("lfq")
@dataclass
class LFQConfig(QuantizerConfig):
    codebook_size: int = 256
    entropy_loss_weight: float = 0.1

    def __post_init__(self):
        power = math.log2(self.codebook_size)
        if power != int(power):
            raise ValueError(
                f"LFQConfig.codebook_size must be a power of 2, got {self.codebook_size}"
            )

    @property
    def codebook_dim(self) -> int:
        return int(math.log2(self.codebook_size))


@QuantizerConfig.register_subclass("noquant")
@dataclass
class NoQuantConfig(QuantizerConfig):
    """Continuous Gaussian latent (ordinary VAE posterior + unit-Gaussian prior).

    ``beta_kl`` multiplies the **per-cell KL** (sum over latent tokens × dims,
    mean over batch).  Under this reduction, ``β_kl · kl_per_cell`` is the
    direct KL contribution per cell measured in nats — readable without
    having to mentally factor out the latent shape.

    Default ``beta_kl=1e-6`` reproduces the previous training dynamics when
    the latent shape was ``(L=32, D=32)`` and reduction was full ``.mean()``
    with ``beta_kl=1e-3`` (since new_β = old_β / (L·D) = 1e-3 / 1024 ≈ 1e-6).
    Raise toward ``1e-5``–``1e-4`` to tighten posterior against the prior
    (useful if you plan to sample from ``N(0, I)`` through the decoder);
    lower to approach a plain autoencoder.
    """

    latent_dim: int = 32
    beta_kl: float = 1e-6

    @property
    def codebook_dim(self) -> int:
        return self.latent_dim
