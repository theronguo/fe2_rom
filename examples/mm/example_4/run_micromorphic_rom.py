"""Standalone micromorphic ROM homogenization demo for the hexagonal RVE.

ROM counterpart of ``run_micromorphic.py``: exercises
:class:`ReducedMicroSolver` (POD + ECM hyper-reduced RVE with micromorphic
enrichment) on the same hexagonal cell and φ-modes as the full-order demo, then
reports ``P̄``, ``Π``, ``Λ`` and the 9-block tangent grid.

Unlike the square RVE of example 1, the porous hexagon needs its exact cell area
``|Q|`` for averaging (the ``∫1 dX`` fallback would equal the *solid* area, not
``|Q|``), so we pass ``rve_volume`` to the solver. The reduced solver does not
enforce periodicity constraints (the POD basis already spans constraint-
satisfying snapshots), so it does not need ``lattice_vectors``.

Prerequisites:
    1. python generate_training_data.py   # → output_gen/{snapshots_pool, modes/phi}
    2. python build_rom.py                 # → ecm/{indices,basis_*,omega_sub}.npy

Run:
    python run_micromorphic_rom.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import logging
import re

import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.rom.solver_mm import ReducedMicroSolver


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.mm.example_4_rom")
logger.addFilter(lambda r: comm.rank == 0)


# --- Paths ------------------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "hexagonal_rve.msh"))
ROM_DIR  = os.path.abspath(os.path.join(HERE, "ecm"))
PHI_DIR  = os.path.abspath(os.path.join(HERE, "output_gen", "modes", "phi"))

output_dir = os.path.join(HERE, "output_rom")
os.makedirs(output_dir, exist_ok=True)

gdim = 2
degree = 2


# --- Hexagon geometry (cell area |Q| for averaging) -------------------------
def hexagon_volume_from_mesh(path):
    """Exact regular-hexagon cell area |Q| = 2√3 ℓ², apothem ℓ from the bbox."""
    ell = None
    if comm.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.open(path)
            xy = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)[:, :2]
        finally:
            gmsh.finalize()
        ell = 0.5 * (xy[:, 0].max() - xy[:, 0].min())
    ell = comm.bcast(ell, root=0)
    return 2.0 * np.sqrt(3.0) * ell ** 2


RVE_VOLUME = hexagon_volume_from_mesh(RVE_MESH)


# --- Material ---------------------------------------------------------------
E_micro, nu_micro = 3000.0, 0.30
mu_micro  = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))


# --- Load φᵢ as raw full-mesh DOF arrays (the solver builds its own spaces). -
phi_files = sorted(
    f for f in glob.glob(os.path.join(PHI_DIR, "phi_*.npy"))
    if re.search(r"phi_[\d.]+\.npy$", os.path.basename(f))
)
if not phi_files:
    raise FileNotFoundError(
        f"No phi_*.npy found in {PHI_DIR}. Run generate_training_data.py first."
    )
phi_arrays = [np.load(fp) for fp in phi_files]
N_MODES = len(phi_arrays)
logger.info("Loaded %d φ mode(s) from %s", N_MODES, PHI_DIR)


# --- Solver -----------------------------------------------------------------
solver = ReducedMicroSolver(
    mesh_path=RVE_MESH,
    rom_dir=ROM_DIR,
    material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
    phi=phi_arrays,
    gdim=gdim,
    degree=degree,
    comm=comm,
    output_dir=output_dir,
    visualize_fields=["u_fluc", "u_total"],
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                    "max_iter": 50, "div_rel_tol": 10.0},
    timestepper_options={"t_end": 1.0, "dt_init": 1e-1, "dt_min": 1e-5,
                         "dt_max": 1e-1, "good_newton_steps": 5},
    rve_volume=RVE_VOLUME,
    averages_only_final=False,
)


# --- Macro inputs (mirrors run_micromorphic.py: probe mode 0) ----------------
Fbar_target = np.array([[0.9, 0.05],
                        [-0.03, 1.0]])
v_target = np.zeros(N_MODES)
v_target[0] = 0.5
v_target[1] = -0.2
v_target[2] = 0.7
g_target = np.zeros((N_MODES, 2))
g_target[0] = (-0.1, 0.05)
g_target[1] = (0.12, -0.03)
g_target[2] = (0.05, 0.02)

logger.info("Calling micromorphic ROM solver with N=%d modes …", N_MODES)
results = solver(Fbar_target, v_target, g_target)
final = results[-1]


# --- Report -----------------------------------------------------------------
if comm.rank == 0:
    logger.info("── Effective quantities (final load) ──")
    for key in ("Fbar", "Pbar", "Pi", "Lambda"):
        if key in final:
            logger.info("  %-8s shape=%s\n%s",
                        key, np.shape(final[key]), np.asarray(final[key]))

    logger.info("── Tangent block norms ──")
    for key in sorted(k for k in final if k.startswith("d")):
        arr = np.asarray(final[key])
        logger.info("  %-15s shape=%s  |.|=%.4e",
                    key, arr.shape, np.linalg.norm(arr))

    np.savez(
        os.path.join(output_dir, "micromorphic_rom_results.npz"),
        **{k: np.asarray(v) for k, v in final.items()},
    )
    logger.info("Saved results to %s/micromorphic_rom_results.npz", output_dir)


# --- Plot history (mode 0) --------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Fbar_hist = np.array([np.asarray(r["Fbar"])       for r in results])
Pbar_hist = np.array([np.asarray(r["Pbar"])       for r in results])
Pi_hist   = np.array([np.asarray(r["Pi"])         for r in results])
Lam_hist  = np.array([np.asarray(r["Lambda"])     for r in results])
A_hist    = np.array([np.asarray(r["dPbar_dFbar"]) for r in results])
dPidv_h   = np.array([np.asarray(r["dPi_dv"])     for r in results])
dLamdg_h  = np.array([np.asarray(r["dLambda_dg"]) for r in results])

F00 = Fbar_hist[:, 0, 0]
denom = (1.0 - Fbar_target[0, 0]) or 1.0
t_ramp = (1.0 - F00) / denom
v0_hist = t_ramp * v_target[0]
gx_hist = t_ramp * g_target[0, 0]

if comm.rank == 0:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].plot(F00, Pbar_hist[:, 0, 0], "-o", ms=3)
    axes[0, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar P_{xx}$")
    axes[0, 1].plot(v0_hist, Pi_hist[:, 0], "-o", ms=3)
    axes[0, 1].set(xlabel=r"$v_0$", ylabel=r"$\Pi_0$")
    axes[0, 2].plot(gx_hist, Lam_hist[:, 0, 0], "-o", ms=3)
    axes[0, 2].set(xlabel=r"$g_{0,x}$", ylabel=r"$\Lambda_{0,x}$")
    axes[1, 0].plot(F00, A_hist[:, 0, 0, 0, 0], "-o", ms=3)
    axes[1, 0].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$\bar A_{xxxx}$")
    axes[1, 1].plot(F00, dPidv_h[:, 0, 0], "-o", ms=3)
    axes[1, 1].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$d\Pi_0/dv_0$")
    axes[1, 2].plot(F00, dLamdg_h[:, 0, 0, 0, 0], "-o", ms=3)
    axes[1, 2].set(xlabel=r"$\bar F_{xx}$", ylabel=r"$d\Lambda_{0,x}/dg_{0,x}$")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "micromorphic_rom_history.png")
    fig.savefig(plot_path, dpi=150)
    logger.info("Saved history plots to %s", plot_path)
