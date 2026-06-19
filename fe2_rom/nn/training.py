"""Lightning training of :class:`~fe2_rom.nn.model.EnergyNet` with a Sobolev
loss, shared across the CH1 / MM / CH2 flavors.

The loss matches the effective energy *and* its first derivatives — by
Hellmann–Feynman the homogenized stresses are exactly the partial derivatives
of W̄ at the converged micro state::

    ch1:  P̄ = ∂W̄/∂F̄
    mm :  P̄ = ∂W̄/∂F̄,   Πᵢ = ∂W̄/∂vᵢ,   Λᵢ = ∂W̄/∂gᵢ
    ch2:  P̄ = ∂W̄/∂F̄,   Q̄  = ∂W̄/∂Ḡ

so every sample contributes ``1 + (flux dimensions)`` scalar targets. Stresses
are what enter the macro residual; matching them directly gives far better
flux/tangent accuracy than an energy-only fit.

Dataset ``.npz`` layout (n samples) by flavor::

    common:  F (n, gdim, gdim)    sampled F̄
             W (n,)               effective energy density W̄
             P (n, gdim, gdim)    effective 1st PK stress P̄
    mm:      v (n, N)             mode amplitudes
             g (n, N, gdim)       mode-gradient amplitudes
             Pi (n, N)            effective Π
             Lambda (n, N, gdim)  effective Λ
    ch2:     G (n, gdim, gdim, gdim)   gradient Ḡ (symmetric in last two indices)
             Q (n, gdim, gdim, gdim)   double stress Q̄

Everything runs in float64 (use ``Trainer(precision="64-true")``) — the
deployed material feeds consistent tangents to SNES.
"""
from __future__ import annotations

import numpy as np
import torch
import lightning as L
from torch.func import grad_and_value, vmap
from torch.utils.data import DataLoader, TensorDataset

from fe2_rom.nn.model import (
    EnergyNet, make_lab_energy, _F_DIM, _F_ORDER, _U_COMPONENTS,
)


# ---------------------------------------------------------------------------
# Flux layout (order of the ∂W/∂z target vector, matching make_lab_energy)
# ---------------------------------------------------------------------------

def flux_spec(model: EnergyNet) -> list[tuple[str, int]]:
    """Ordered ``[(name, dim), …]`` split of the gradient target ∂W/∂z.

    Zero-dimensional fluxes (e.g. mm with ``n_modes == 0``) are dropped.
    """
    F_dim, gdim, N = _F_DIM[model.gdim], model.gdim, model.n_modes
    if model.flavor == "ch1":
        spec = [("P", F_dim)]
    elif model.flavor == "mm":
        spec = [("P", F_dim), ("Pi", N), ("Lambda", N * gdim)]
    elif model.flavor == "ch2":
        spec = [("P", F_dim), ("Q", gdim ** 3)]
    else:  # pragma: no cover — EnergyNet validates flavor
        raise ValueError(model.flavor)
    return [(name, dim) for name, dim in spec if dim > 0]


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------

def _mat_to_fvec_batch(T: torch.Tensor, gdim: int) -> torch.Tensor:
    """(n, gdim, gdim) → (n, F_dim) in MFront ordering (zero placeholder)."""
    n = T.shape[0]
    out = torch.zeros(n, _F_DIM[gdim], dtype=T.dtype)
    for k, ij in enumerate(_F_ORDER[gdim]):
        if ij is not None:
            out[:, k] = T[:, ij[0], ij[1]]
    return out


def pack_dataset(data: dict, model: EnergyNet) -> tuple[torch.Tensor, ...]:
    """Arrays → (z, W, dWdz) tensors in the MFront packing of make_lab_energy."""
    gdim, flavor = model.gdim, model.flavor
    as_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    F = as_t(data["F"])
    n = F.shape[0]
    z_parts = [_mat_to_fvec_batch(F, gdim)]
    g_parts = [_mat_to_fvec_batch(as_t(data["P"]), gdim)]
    if flavor == "mm":
        z_parts += [as_t(data["v"]), as_t(data["g"]).reshape(n, -1)]
        g_parts += [as_t(data["Pi"]), as_t(data["Lambda"]).reshape(n, -1)]
    elif flavor == "ch2":
        z_parts += [as_t(data["G"]).reshape(n, -1)]
        g_parts += [as_t(data["Q"]).reshape(n, -1)]
    z = torch.cat(z_parts, dim=1)
    dWdz = torch.cat(g_parts, dim=1)
    return z, as_t(data["W"]), dWdz


def reduced_coords(z: torch.Tensor, model: EnergyNet) -> torch.Tensor:
    """z (lab packing) → x (model input) for input-standardisation calibration.

    Uses the symmetric-stretch approximation (R ≈ I, so Ū = F̄ components and
    Ĝ = Ḡ) — this only sizes the per-input mean/std, so the approximation is
    harmless; the loss itself uses the exact polar reduction in make_lab_energy.
    """
    gdim, F_dim = model.gdim, _F_DIM[model.gdim]
    order = _F_ORDER[gdim]
    cols = [z[:, order.index((i, j))] for (i, j) in _U_COMPONENTS[gdim]]
    return torch.cat([torch.stack(cols, dim=1), z[:, F_dim:]], dim=1)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class EnergyDataModule(L.LightningDataModule):
    """Loads a dataset .npz, packs MFront vectors, splits train/val."""

    def __init__(self, npz_path: str, model: EnergyNet, batch_size: int = 256,
                 val_fraction: float = 0.1, seed: int = 0):
        super().__init__()
        self.npz_path = npz_path
        self.model = model
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.scales: dict | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.scales is not None:
            return
        with np.load(self.npz_path) as data:
            z, W, dWdz = pack_dataset(data, self.model)
        n = z.shape[0]
        rng = np.random.default_rng(self.seed)
        perm = torch.as_tensor(rng.permutation(n))
        n_val = max(1, int(round(self.val_fraction * n)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self._train = TensorDataset(z[train_idx], W[train_idx], dWdz[train_idx])
        self._val = TensorDataset(z[val_idx], W[val_idx], dWdz[val_idx])

        # Per-target scales (from the training split) so loss weights are O(1).
        G = dWdz[train_idx]
        std = lambda t: max(float(t.std()), 1e-12)
        scales = {"W": std(W[train_idx])}
        off = 0
        for name, dim in flux_spec(self.model):
            scales[name] = std(G[:, off:off + dim])
            off += dim
        self.scales = scales
        self._train_x = reduced_coords(z[train_idx], self.model)
        self._train_W = W[train_idx]

    def attach_normalization(self, model: EnergyNet) -> None:
        """Calibrate the model's input standardisation and output scale."""
        self.setup()
        x = self._train_x
        std = x.std(dim=0).clamp_min(1e-8)
        model.x_mean.copy_(x.mean(dim=0))
        model.x_std.copy_(std)
        model.W_scale.fill_(max(float(self._train_W.std()), 1e-12))

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class EnergyModule(L.LightningModule):
    """Sobolev training: ``w_W·MSE(W̄) + Σ_flux w_flux·MSE(flux)``, each target
    normalised by its training-set std.

    Parameters
    ----------
    model : EnergyNet
    lr : float
    weights : dict, optional
        Per-target loss weights, e.g. ``{"W": 1.0, "P": 1.0, "Q": 1.0}``;
        missing entries default to 1.0.
    scales : dict, optional
        Per-target normalisation (typically ``EnergyDataModule.scales``).
    """

    def __init__(self, model: EnergyNet, lr: float = 1e-3,
                 weights: dict | None = None, scales: dict | None = None):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self._spec = flux_spec(model)
        names = ["W"] + [name for name, _ in self._spec]
        weights = weights or {}
        scales = scales or {}
        self.register_buffer("_weights", torch.tensor(
            [float(weights.get(k, 1.0)) for k in names], dtype=torch.float64))
        self.register_buffer("_scales", torch.tensor(
            [float(scales.get(k, 1.0)) for k in names], dtype=torch.float64))
        self._names = names

    def _losses(self, batch):
        z, W_t, G_t = batch
        # Keep the parameter dependence of the reference correction so the
        # optimiser trains exactly the deployed (corrected) model.
        ref = self.model.reference_terms(create_graph=True)
        W_fn = make_lab_energy(self.model, ref)
        G_p, W_p = vmap(grad_and_value(W_fn))(z)

        mse = lambda a, b, s: torch.mean(((a - b) / s) ** 2)
        loss_W = mse(W_p, W_t, self._scales[0])
        total = self._weights[0] * loss_W
        parts = {"W": loss_W}
        off = 0
        for k, (name, dim) in enumerate(self._spec, start=1):
            sl = slice(off, off + dim)
            off += dim
            li = mse(G_p[:, sl], G_t[:, sl], self._scales[k])
            total = total + self._weights[k] * li
            parts[name] = li
        return total, parts

    def training_step(self, batch, batch_idx):
        total, parts = self._losses(batch)
        self.log("train_loss", total, prog_bar=True)
        for k, val in parts.items():
            self.log(f"train_{k}", val)
        return total

    def validation_step(self, batch, batch_idx):
        # Sobolev loss needs autograd through the model — re-enable grad.
        with torch.enable_grad():
            total, parts = self._losses(batch)
        self.log("val_loss", total, prog_bar=True)
        for k, val in parts.items():
            self.log(f"val_{k}", val)
        return total

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=50)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
