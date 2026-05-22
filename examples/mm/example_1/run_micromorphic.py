"""Standalone micromorphic homogenization demo.

Exercises :class:`MicroSolver` on the
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

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.mm.microsolver import MicroSolver


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
solver = MicroSolver(
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
    timestepper_options={"t_end": 1.0, "dt_init": 1e-2, "dt_min": 1e-5,
                         "dt_max": 1e-2, "good_newton_steps": 5},
    averages_only_final=False,
    save_snapshots=["u_fluc", "P"],
)


# --- Populate φᵢ from a linear buckling analysis ---------------------------
# The smallest-|λ| eigenmodes of K at the reference state are used as the
# global modes. With Fbar=I (default) K is SPD and you get the softest
# deformation modes; pass an Fbar closer to the critical load to recover
# true buckling modes.
eigvals = solver.compute_linear_buckling_modes(N_MODES, visualize_modes=True, save_modes=True)
logger.info("Buckling eigenvalues: %s", eigvals)


# --- Macro inputs -----------------------------------------------------------
Fbar_target = np.array([[0.9, 0.0],
                        [0.0,  1.0]])
v_target = np.array([1])
g_target = np.array([[-0.2, 0.1]])

logger.info("Calling micromorphic solver with N=%d modes …", N_MODES)
results = solver(Fbar_target, v_target, g_target)
final = results[-1]


# --- Constraint violation check ---------------------------------------------
w_vec = solver.u.x.petsc_vec
c_vecs = solver._newton._constraint_vecs
if c_vecs:
    viols = [abs(c.dot(w_vec)) for c in c_vecs]
    logger.info("Constraint violations (should be ~0): max=%.3e  per-row=%s",
                max(viols), np.array2string(np.array(viols), precision=2))
else:
    logger.info("No constraints active.")


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


import matplotlib.pyplot as plt

# Recover the load ramp from Fbar history: t = (1 - F[0,0]) / (1 - F_target[0,0]).
Fbar_hist  = np.array([np.asarray(r["Fbar"])     for r in results])
Pbar_hist  = np.array([np.asarray(r["Pbar"])     for r in results])
Pi_hist    = np.array([np.asarray(r["Pi"])       for r in results])
Lam_hist   = np.array([np.asarray(r["Lambda"])   for r in results])
A_hist     = np.array([np.asarray(r["dPbar_dFbar"])  for r in results])
dPidv_h    = np.array([np.asarray(r["dPi_dv"])       for r in results])
dLamdg_h   = np.array([np.asarray(r["dLambda_dg"])   for r in results])

F00 = Fbar_hist[:, 0, 0]
denom = (1.0 - Fbar_target[0, 0]) or 1.0
t_ramp = (1.0 - F00) / denom
v0_hist = t_ramp * v_target[0]
gx_hist = t_ramp * g_target[0, 0]

fig, axes = plt.subplots(2, 3, figsize=(13, 7))

axes[0, 0].plot(F00, Pbar_hist[:, 0, 0], "-o", ms=3)
axes[0, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar P_{xx}$",
                title=r"$\bar P_{xx}$ vs $\bar F_{xx}$")

axes[0, 1].plot(v0_hist, Pi_hist[:, 0], "-o", ms=3)
axes[0, 1].set(xlabel=r"$v_0$", ylabel=r"$\Pi_0$",
                title=r"$\Pi$ vs $v$")

axes[0, 2].plot(gx_hist, Lam_hist[:, 0, 0], "-o", ms=3)
axes[0, 2].set(xlabel=r"$g_{0,x}$", ylabel=r"$\Lambda_{0,x}$",
                title=r"$\Lambda_x$ vs $g_x$")

axes[1, 0].plot(F00, A_hist[:, 0, 0, 0, 0], "-o", ms=3)
axes[1, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar A_{xxxx}$",
                title=r"$\bar A_{xxxx}$ vs $\bar F_{xx}$")

axes[1, 1].plot(F00, dPidv_h[:, 0, 0], "-o", ms=3)
axes[1, 1].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$d\Pi_0/dv_0$",
                title=r"$d\Pi/dv$ vs $\bar F_{xx}$")

axes[1, 2].plot(F00, dLamdg_h[:, 0, 0, 0, 0], "-o", ms=3)
axes[1, 2].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$d\Lambda_{0,x}/dg_{0,x}$",
                title=r"$d\Lambda_x/dg_x$ vs $\bar F_{xx}$")

for ax in axes.ravel():
    ax.grid(True, alpha=0.3)

fig.tight_layout()
plot_path = os.path.join(output_dir, "micromorphic_history.png")
fig.savefig(plot_path, dpi=150)
logger.info("Saved history plots to %s", plot_path)
