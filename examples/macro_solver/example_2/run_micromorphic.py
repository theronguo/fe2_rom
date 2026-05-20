"""Standalone micromorphic homogenization demo.

Exercises :class:`MicromorphicHyperelasticHomogenizationSolver` on the
2D RVE mesh from ``examples/periodic_solver/example_1`` with a single
user-supplied global mode ``φ₀`` (a smooth periodic sinusoid). Reports
``Pbar``, ``Π``, ``Λ`` and the full 9-block tangent grid at the final
load state.

Constraints on ``w`` beyond corner-pinning are not enforced — this is the
"ignore constraints" path of Stage B.

Run:
    conda activate fe2_rom_env
    python run_micromorphic.py
"""
import os
# Match run_macro.py: keep nested solves single-threaded to avoid oversubscription.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import (
    MicromorphicHyperelasticHomogenizationSolver,
    NeoHookean,
    setup_logging,
)


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.macro_solver.example_2")
logger.addFilter(lambda r: comm.rank == 0)


# --- RVE setup --------------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(
    os.path.join(HERE, "rve.msh")
)

E_micro, nu_micro = 3000.0, 0.30
mu_micro = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))

output_dir = os.path.join(HERE, "output")
os.makedirs(output_dir, exist_ok=True)


# --- Solver ---------------------------------------------------------------
N_MODES = 1
solver = MicromorphicHyperelasticHomogenizationSolver(
    mesh_path=RVE_MESH,
    comm=comm,
    gdim=2,
    material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
    N=N_MODES,
    degree=2,
    output_dir=output_dir,
    check_stability=True,
    visualize_fields=["u_fluc", "u_total", "F", "P"],
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                    "max_iter": 50, "div_rel_tol": 10,
                    "switch_to_minres": True},
    timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5,
                         "dt_max": 1.0, "good_newton_steps": 5},
    averages_only_final=True
)


# --- Populate φᵢ from a linear buckling analysis ---------------------------
# The smallest-|λ| eigenmodes of K at the reference state are used as the
# global modes. With Fbar=I (default) K is SPD and you get the softest
# deformation modes; pass an Fbar closer to the critical load to recover
# true buckling modes.
eigvals = solver.compute_linear_buckling_modes(N_MODES, visualize_modes=True)
logger.info("Buckling eigenvalues: %s", eigvals)


# --- Macro inputs -----------------------------------------------------------
Fbar_target = np.array([[0.9, 0.0],
                        [0.0,  1.0]])
v_target = np.array([1])
g_target = np.array([[-0.2, 0.1]])

logger.info("Calling micromorphic solver with N=%d modes …", N_MODES)
results = solver(Fbar_target, v_target, g_target, pert_amplitude_init=0.1)
final = results[-1]


# --- Report -----------------------------------------------------------------
if comm.rank == 0:
    logger.info("── Effective quantities (final load) ──")
    for key in ("Fbar", "Pbar", "Pi", "Lambda"):
        logger.info("  %-8s shape=%s\n%s",
                    key, np.shape(final[key]), np.asarray(final[key]))

    logger.info("── Tangent block norms ──")
    for key in sorted(k for k in final if k.startswith("d")):
        arr = np.asarray(final[key])
        logger.info("  %-15s shape=%s  |.|=%.4e",
                    key, arr.shape, np.linalg.norm(arr))

    np.savez(
        os.path.join(output_dir, "micromorphic_results.npz"),
        **{k: np.asarray(v) for k, v in final.items()},
    )
    logger.info("Saved results to %s/micromorphic_results.npz", output_dir)
