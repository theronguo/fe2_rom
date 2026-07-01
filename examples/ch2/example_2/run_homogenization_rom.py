"""Second-order (CH2) reduced-order online homogenization on the 2D square RVE.

ROM counterpart of ``run_homogenization.py`` (and the CH2 analogue of
``examples/mm/example_1/run_micromorphic_rom.py``): drives the POD + ECM reduced
RVE built by ``build_rom.py`` (read from ``ecm/``) with the *same* macroscopic
load ``(F̄, Ḡ)`` as the full-order run, and writes the reduced response to
``output_rom/`` for a side-by-side comparison.

Prerequisites:
    1. python generate_training_data.py   # (F̄, Ḡ) snapshot pool → output_gen/
    2. python build_rom.py                 # POD + ECM → ecm/

Run (serial only):
    python run_homogenization_rom.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.rom.solver_ch2 import ReducedMicroSolver


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.ch2.example_2_rom")
logger.addFilter(lambda r: comm.rank == 0)

# --- Paths ------------------------------------------------------------------
# Set FULL_QUADRATURE=True to drive the ROM with the exact-cubature model from
# `ecm_full/` (build it via build_rom.py with FULL_QUADRATURE=True). Its
# FOM-vs-ROM error is pure POD (Galerkin) error; the difference from the `ecm/`
# run is the ECM (hyper-reduction) contribution.
FULL_QUADRATURE = False
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "rve.msh"))
ROM_DIR = os.path.abspath(os.path.join(HERE, "ecm_full" if FULL_QUADRATURE else "ecm"))
output_dir = os.path.join(HERE, "output_rom_full" if FULL_QUADRATURE else "output_rom")
os.makedirs(output_dir, exist_ok=True)

# --- Material ---------------------------------------------------------------
E_micro, nu_micro = 3000.0, 0.30
mu_micro = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))
GDIM = 2
DEGREE = 2


# --- Reduced second-order solver --------------------------------------------
solver = ReducedMicroSolver(
    mesh_path=RVE_MESH,
    rom_dir=ROM_DIR,
    material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
    gdim=GDIM,
    degree=DEGREE,
    comm=comm,
    output_dir=output_dir,
    visualize_fields=["u_fluc", "u_total"],
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                    "max_iter": 50, "div_rel_tol": 10.0},
    timestepper_options={"t_end": 1.0, "dt_init": 1e-2, "dt_min": 1e-5,
                         "dt_max": 1e-2, "good_newton_steps": 5},
    averages_only_final=False,
)


# --- Macro inputs: identical to run_homogenization.py -----------------------
Fbar_target = np.array([[0.92, 0.0],
                        [0.0,  0.98]])
Gbar_target = np.zeros((GDIM, GDIM, GDIM))
Gbar_target[0, 0, 0] = 0.04     # ∂F̄_xx/∂X_x
Gbar_target[1, 1, 1] = -0.03    # ∂F̄_yy/∂X_y

logger.info("CH2 ROM solve: F̄_xx=%.3f, F̄_yy=%.3f, Ḡ_xxx=%.3f, Ḡ_yyy=%.3f …",
            Fbar_target[0, 0], Fbar_target[1, 1],
            Gbar_target[0, 0, 0], Gbar_target[1, 1, 1])
results = solver(Fbar_target, Gbar_target)
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
    np.savez(os.path.join(output_dir, "ch2_rom_results.npz"),
             **{k: np.asarray(v) for k, v in final.items()})
    logger.info("Saved results to %s/ch2_rom_results.npz", output_dir)

    # --- FOM vs ROM comparison (if the FOM run is available) ----------------
    fom_path = os.path.join(HERE, "output", "ch2_results.npz")
    if os.path.exists(fom_path):
        fom = np.load(fom_path)
        def rel(a, b):
            b = np.asarray(b)
            denom = np.linalg.norm(b) or 1.0
            return np.linalg.norm(np.asarray(a) - b) / denom
        logger.info("── FOM vs ROM (relative error) ──")
        for key in ("Pbar", "Qbar", "dPbar_dFbar", "dQbar_dG"):
            if key in fom.files and key in final:
                logger.info("  %-12s  rel.err = %.3e", key, rel(final[key], fom[key]))


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
    plot_path = os.path.join(output_dir, "ch2_rom_history.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved history plots to %s", plot_path)
