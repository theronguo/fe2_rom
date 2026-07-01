"""End-to-end second-order (CH2) homogenization on the 2D square RVE.

Full-order counterpart of ``run_homogenization_rom.py`` and the CH2 analogue of
``examples/mm/example_1/run_micromorphic.py``. Drives the periodic RVE with a
macroscopic deformation gradient ``F̄`` *and* its gradient ``Ḡ`` via the
strain-gradient ansatz (paper Eq. 10)

    u_total = (F̄ − I)·X + ½ X·Ḡ·X + w(X),

and reports the effective stress ``P̄`` (Eq. 26), the double stress ``Q̄``
(Eq. 27) and the four macro tangents ``d{P̄, Q̄}/d{F̄, Ḡ}`` at every accepted
load step. A zero ``Ḡ`` recovers the first-order (ch1) response exactly.

Run:
    conda activate fe2_rom_env
    python run_homogenization.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.ch2 import MicroSolver


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.ch2.example_2")
logger.addFilter(lambda r: comm.rank == 0)

# --- RVE + material ---------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "rve.msh"))
output_dir = os.path.join(HERE, "output")
os.makedirs(output_dir, exist_ok=True)

E_micro, nu_micro = 3000.0, 0.30
mu_micro = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))
material = NeoHookean(mu=mu_micro, lmbda=lam_micro)

LATTICE_VECTORS = None      # axis-aligned box RVE (centred at the origin)
GDIM = 2
DEGREE = 2


# --- Full-order second-order solver -----------------------------------------
solver = MicroSolver(
    mesh_path=RVE_MESH, comm=comm, gdim=GDIM, material=material, degree=DEGREE,
    output_dir=output_dir, check_stability=True, perturb_post_buckling=True,
    visualize_fields=["u_fluc", "u_total", "F", "P"],
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                    "max_iter": 50, "div_rel_tol": 10, "switch_to_minres": True},
    timestepper_options={"t_end": 1.0, "dt_init": 1e-2, "dt_min": 1e-5,
                         "dt_max": 1e-2, "good_newton_steps": 5},
    lattice_vectors=LATTICE_VECTORS,
)


# --- Macro inputs: F̄ and its gradient Ḡ (symmetric in J↔K) ------------------
Fbar_target = np.array([[0.92, 0.0],
                        [0.0,  0.98]])
Gbar_target = np.zeros((GDIM, GDIM, GDIM))
Gbar_target[0, 0, 0] = 0.04     # ∂F̄_xx/∂X_x
Gbar_target[1, 1, 1] = -0.03    # ∂F̄_yy/∂X_y

logger.info("CH2 solve: F̄_xx=%.3f, F̄_yy=%.3f, Ḡ_xxx=%.3f, Ḡ_yyy=%.3f …",
            Fbar_target[0, 0], Fbar_target[1, 1],
            Gbar_target[0, 0, 0], Gbar_target[1, 1, 1])
results = solver(Fbar_target, Gbar_target, pert_amplitude_init=1e-3)
final = results[-1]


# --- Report -----------------------------------------------------------------
if comm.rank == 0:
    logger.info("── Effective quantities (final load) ──")
    for key in ("Fbar", "Pbar", "Qbar"):
        if key in final:
            logger.info("  %-8s shape=%s\n%s",
                        key, np.shape(final[key]), np.asarray(final[key]))
    logger.info("── Tangent block norms ──")
    for key in sorted(k for k in final if k.startswith("d")):
        arr = np.asarray(final[key])
        logger.info("  %-15s shape=%s  |.|=%.4e", key, arr.shape, np.linalg.norm(arr))
    np.savez(os.path.join(output_dir, "ch2_results.npz"),
             **{k: np.asarray(v) for k, v in final.items()})
    logger.info("Saved results to %s/ch2_results.npz", output_dir)


# --- History plots ----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Fbar_hist = np.array([np.asarray(r["Fbar"]) for r in results])
Pbar_hist = np.array([np.asarray(r["Pbar"]) for r in results])
Qbar_hist = np.array([np.asarray(r["Qbar"]) for r in results])
A_hist = np.array([np.asarray(r["dPbar_dFbar"]) for r in results])

Fxx = Fbar_hist[:, 0, 0]

if comm.rank == 0:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(Fxx, Pbar_hist[:, 0, 0], "-o", ms=3)
    axes[0, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar P_{xx}$")
    axes[0, 1].plot(Fxx, Pbar_hist[:, 1, 1], "-o", ms=3)
    axes[0, 1].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar P_{yy}$")
    axes[1, 0].plot(Fxx, Qbar_hist[:, 0, 0, 0], "-o", ms=3)
    axes[1, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar Q_{xxx}$")
    axes[1, 1].plot(Fxx, A_hist[:, 0, 0, 0, 0], "-o", ms=3)
    axes[1, 1].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar A_{xxxx}$")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "ch2_history.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved history plots to %s", plot_path)
