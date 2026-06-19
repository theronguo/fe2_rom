"""Validation of the CANN (Linka et al., JCP 429 (2021) 110010) against the
Mooney–Rivlin reference of section 4.1.1.

Reproduces the protocol of section 4.1 / Appendix B (Table B1):

* training data: the generalized Mooney–Rivlin energy (18) with the Table-1
  parameters, sampled along uniaxial tension (UT), equi-biaxial tension (EBT)
  and pure shear (PS), 15 points per load case, inside the I_C–II_C triangle
  with corners (3, 3), (50, 15), (32, 256);
* model: isotropic incompressible CANN (R = 1, L̄₁ = ⅓I, inputs Ī₁/J̄₁, III_C
  and the feature vector removed) with the Fig. B2 architecture;
* training: Adam (lr 1e-3, β=(0.9, 0.999), ε=1e-7), Glorot init, MSE on the 2nd
  PK stress S (eq. B1), 80/20 split, 4000 epochs, batch 4, best epoch kept.

Run inside the dolfinx-rve container:
    conda activate fe2_rom_env && python validation/cann/validate_mooney_rivlin.py
"""
import fe2_rom  # noqa: F401  (dolfinx-before-torch import order)
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cann import CANN  # noqa: E402  (local: validation/cann/cann.py)

torch.manual_seed(0)
np.random.seed(0)
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# Mooney–Rivlin parameters (Table 1) [MPa]
C10, C20, C30 = 1.6e-1, -1.4e-3, 3.9e-5
C01, C02, C03 = 1.5e-2, -2.0e-6, 1.0e-10


# ---------------------------------------------------------------------------
# Mooney–Rivlin reference (eq. 18)
# ---------------------------------------------------------------------------
def mr_energy(Ic, IIc):
    a, b = Ic - 3.0, IIc - 3.0
    return (C10 * a + C20 * a**2 + C30 * a**3
            + C01 * b + C02 * b**2 + C03 * b**3)


def mr_dWdI(Ic, IIc):
    a, b = Ic - 3.0, IIc - 3.0
    W1 = C10 + 2 * C20 * a + 3 * C30 * a**2
    W2 = C01 + 2 * C02 * b + 3 * C03 * b**2
    return W1, W2


def invariants(C):
    Ic = torch.einsum("...ii->...", C)
    trC2 = torch.einsum("...ij,...ji->...", C, C)
    IIc = 0.5 * (Ic**2 - trC2)
    return Ic, IIc


def mr_stress_S(C):
    """Constitutive 2nd PK S = 2 ∂Ψ_MR/∂C (no pressure)."""
    Ic, IIc = invariants(C)
    W1, W2 = mr_dWdI(Ic, IIc)
    eye = torch.eye(3, dtype=C.dtype).expand_as(C)
    return 2.0 * ((W1 + Ic * W2)[..., None, None] * eye - W2[..., None, None] * C)


# ---------------------------------------------------------------------------
# Incompressible homogeneous deformations
# ---------------------------------------------------------------------------
def F_of(load, lam):
    if load == "UT":
        return np.diag([lam, lam**-0.5, lam**-0.5])
    if load == "EBT":
        return np.diag([lam, lam, lam**-2])
    if load == "PS":
        return np.diag([lam, 1.0, lam**-1])
    raise ValueError(load)


LOADS = {"UT": (1.0, 7.0), "EBT": (1.0, 4.0), "PS": (1.0, 6.5)}


def make_dataset(n_per_case=15):
    Cs = []
    for load, (lo, hi) in LOADS.items():
        for lam in np.linspace(lo, hi, n_per_case):
            F = F_of(load, lam)
            Cs.append(F.T @ F)
    C = torch.tensor(np.stack(Cs), dtype=torch.float64)
    return C, mr_stress_S(C)


# ---------------------------------------------------------------------------
# Stress–stretch with pressure elimination (incompressible, principal axes)
# ---------------------------------------------------------------------------
def nominal_P1(s_diag_fn, load, lam):
    """Nominal stress P₁ along the loading axis.

    ``s_diag_fn(C) -> (3,) constitutive S_ii``; pressure from the stress-free
    lateral axis (UT: 2/3, EBT: 3, PS: 3). σ_i = λ_i² s_i − p, P₁ = σ₁/λ₁.
    """
    F = F_of(load, lam)
    lams = np.diag(F)
    C = torch.tensor(F.T @ F, dtype=torch.float64)
    s = s_diag_fn(C)
    lat = {"UT": 1, "EBT": 2, "PS": 2}[load]      # index of a free lateral axis
    p = lams[lat] ** 2 * s[lat]
    sigma1 = lams[0] ** 2 * s[0] - p
    return sigma1 / lams[0]


# ---------------------------------------------------------------------------
# Invariant-plane sampling (map (I_C, II_C) → incompressible C)
# ---------------------------------------------------------------------------
def C_from_invariants(Ic, IIc):
    """Principal stretches λ_i² are the roots of x³ − I_C x² + II_C x − 1."""
    r = np.roots([1.0, -Ic, IIc, -1.0])
    if np.max(np.abs(r.imag)) > 1e-7 or np.min(r.real) <= 0:
        return None
    return np.diag(np.sort(r.real))


def triangle_grid(n=60):
    """Admissible (I_C, II_C) inside the triangle (3,3)-(50,15)-(32,256)."""
    verts = np.array([[3.0, 3.0], [50.0, 15.0], [32.0, 256.0]])
    pts, Cs = [], []
    for Ic in np.linspace(3, 50, n):
        for IIc in np.linspace(3, 256, n):
            # barycentric inside-triangle test
            T = np.array([verts[1] - verts[0], verts[2] - verts[0]]).T
            uv = np.linalg.solve(T, np.array([Ic, IIc]) - verts[0])
            if uv[0] < -1e-9 or uv[1] < -1e-9 or uv.sum() > 1 + 1e-9:
                continue
            Cmat = C_from_invariants(Ic, IIc)
            if Cmat is None:
                continue
            pts.append([Ic, IIc]); Cs.append(Cmat)
    return np.array(pts), torch.tensor(np.stack(Cs), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Training (Table B1)
# ---------------------------------------------------------------------------
def train(model, C, S, epochs=4000, batch=4, lr=1e-3, val_frac=0.2, seed=0):
    n = C.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    vi, ti = perm[:n_val], perm[n_val:]
    Ct, St, Cv, Sv = C[ti], S[ti], C[vi], S[vi]
    model.set_input_norm(Ct)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)

    def mse_S(Cb, Sb):
        return ((model.stress_S(Cb) - Sb) ** 2).sum(dim=(-1, -2)).mean()

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(len(ti)))
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            opt.zero_grad()
            loss = mse_S(Ct[idx], St[idx])
            loss.backward()
            opt.step()
        model.eval()
        tr = mse_S(Ct, St).item()   # stress_S re-enables grad internally
        vl = mse_S(Cv, Sv).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep % 400 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  train {tr:.3e}  val {vl:.3e}  best {best_val:.3e}")
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def rel_l2(pred, ref):
    return float(torch.linalg.norm((pred - ref).reshape(-1))
                 / torch.linalg.norm(ref.reshape(-1)))


def main():
    C, S = make_dataset(15)
    print(f"dataset: {C.shape[0]} points (UT/EBT/PS × 15)")
    model = CANN(structure_tensors=None, hidden=(8, 8), incompressible=True)
    print(f"CANN: R={model.R}, n_inv={model.n_inv}, "
          f"params={sum(p.numel() for p in model.parameters())}")
    train(model, C, S)
    model.eval()

    # --- error across the admissible invariant-plane triangle ---
    pts, Cgrid = triangle_grid(60)
    S_cann = model.stress_S(Cgrid, create_graph=False).detach()
    S_ref = mr_stress_S(Cgrid)
    err_S = rel_l2(S_cann, S_ref)
    # per-point relative stress error (Fig 3d analogue)
    num = torch.linalg.norm((S_cann - S_ref).reshape(len(pts), -1), dim=1)
    den = torch.linalg.norm(S_ref.reshape(len(pts), -1), dim=1).clamp_min(1e-12)
    relP = (num / den).numpy() * 100

    # energy error (offset so Ψ(C=I)=0 for both; MR is already 0 there)
    eye = torch.eye(3, dtype=torch.float64)[None]
    psi0 = model.energy(eye).item()
    psi_cann = (model.energy(Cgrid) - psi0).detach()
    Ic, IIc = invariants(Cgrid)
    psi_ref = mr_energy(Ic, IIc)
    err_W = rel_l2(psi_cann, psi_ref)

    print(f"\nrelative L2 error over invariant-plane triangle ({len(pts)} pts):")
    print(f"  stress S : {err_S*100:.3f} %   (per-point max {relP.max():.3f} %)")
    print(f"  energy Ψ : {err_W*100:.3f} %")

    # --- stress-stretch curves with pressure elimination ---
    def cann_s_diag(Cmat):
        return model.stress_S(Cmat[None], create_graph=False)[0].diagonal().detach().numpy()

    def mr_s_diag(Cmat):
        return mr_stress_S(Cmat[None])[0].diagonal().numpy()

    curves = {}
    for load, (lo, hi) in LOADS.items():
        lams = np.linspace(lo, hi, 60)
        p_cann = [nominal_P1(cann_s_diag, load, l) for l in lams]
        p_mr = [nominal_P1(mr_s_diag, load, l) for l in lams]
        curves[load] = (lams, np.array(p_mr), np.array(p_cann))
        e = np.linalg.norm(np.array(p_cann) - np.array(p_mr)) / np.linalg.norm(p_mr)
        print(f"  {load}: P₁ stress-stretch rel-L2 {e*100:.3f} %")

    _plot(pts, relP, curves)


def _plot(pts, relP, curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(matplotlib unavailable: {exc}; skipping plots)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for load, (lam, mr, cann) in curves.items():
        ax[0].plot(lam, mr, "-", lw=2, label=f"{load} Mooney–Rivlin")
        ax[0].plot(lam, cann, "--", lw=1.6, label=f"{load} CANN")
    ax[0].set_xlabel(r"stretch $\lambda$"); ax[0].set_ylabel(r"nominal stress $P_1$ [MPa]")
    ax[0].legend(fontsize=7); ax[0].set_title("Stress–stretch (Fig. 2/3a)")
    sc = ax[1].scatter(pts[:, 0], pts[:, 1], c=relP, s=14, cmap="viridis")
    fig.colorbar(sc, ax=ax[1], label=r"rel. error in $S$ [%]")
    ax[1].set_xlabel(r"$I_C$"); ax[1].set_ylabel(r"$II_C$")
    ax[1].set_title("Error over invariant plane (Fig. 3d)")
    fig.tight_layout()
    path = os.path.join(OUT, "cann_mooney_rivlin_validation.png")
    fig.savefig(path, dpi=140)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
