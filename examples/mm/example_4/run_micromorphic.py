"""End-to-end micromorphic homogenization on the porous *hexagonal* RVE.

Unlike the square RVE of examples/mm/example_1, this cell is an arbitrary 2D
periodic polygon, so periodicity is enforced through the lattice-vector path of
:class:`fe2_rom.ch1.MicroSolver` (``lattice_vectors=...``).

Pipeline:
  1. extract_buckling_modes — find the enrichment modes φ from full-order RVE
     buckling (equal-biaxial compression, "lba" strategy). The hexagon's
     symmetric bifurcation gives the degenerate triplet (one pattern at
     0°/60°/120°), so N = 3.
  2. build the micromorphic MicroSolver with that many modes and load φ.
  3. micromorphic probe: prescribe (F̄, v, g); report P̄, Π, Λ and the 3×3 grid
     of tangents, plus history plots.

Run:
    conda activate fe2_rom_env
    python run_micromorphic.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.mm import MicroSolver, extract_buckling_modes


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.mm.example_4")
logger.addFilter(lambda r: comm.rank == 0)

# --- Mesh + periodicity -----------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "hexagonal_rve.msh"))
output_dir = os.path.join(HERE, "output")
os.makedirs(output_dir, exist_ok=True)


def hexagon_lattice_from_mesh(path):
    """Infer the pointy-top hexagon geometry from the mesh bounding box and
    return ``(lattice_vectors, cell_area)``. a1 = (2ℓ, 0), a2 = (ℓ, √3 ℓ),
    apothem ℓ; regular-hexagon area |Q| = 2√3 ℓ²."""
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
    s3 = np.sqrt(3.0)
    lattice = np.array([[2.0 * ell, 0.0], [ell, s3 * ell]])
    area = 2.0 * s3 * ell ** 2
    logger.info("Hexagon: apothem ℓ=%.6f, |Q|=%.6f", ell, area)
    return lattice, area


LATTICE_VECTORS, RVE_VOLUME = hexagon_lattice_from_mesh(RVE_MESH)

# --- Material ---------------------------------------------------------------
E_micro, nu_micro = 3000.0, 0.30
mu_micro = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))
material = NeoHookean(mu=mu_micro, lmbda=lam_micro)


# --- 1. Extract the enrichment modes φ from RVE buckling --------------------
res = extract_buckling_modes(
    RVE_MESH, comm, gdim=2, material=material, degree=2,
    lattice_vectors=LATTICE_VECTORS,
    strategy="lba", max_strain=0.10,        # equal-biaxial compression
    output_dir=os.path.join(output_dir, "modes"),
)
N_MODES = res.n_modes
logger.info("Extracted N=%d φ mode(s)", N_MODES)


# --- 2. Build the micromorphic solver and load φ ----------------------------
solver = MicroSolver(
    mesh_path=RVE_MESH, comm=comm, gdim=2, material=material, N=N_MODES, degree=2,
    output_dir=output_dir, check_stability=True,
    visualize_fields=["u_fluc", "u_total", "F", "P"],
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                    "max_iter": 50, "div_rel_tol": 10, "switch_to_minres": True},
    timestepper_options={"t_end": 1.0, "dt_init": 1e-1, "dt_min": 1e-5,
                         "dt_max": 1e-1, "good_newton_steps": 5},
    rve_volume=RVE_VOLUME,
    lattice_vectors=LATTICE_VECTORS,
)
solver.load_buckling_modes([res.phi[:, i] for i in range(N_MODES)])


# --- 3. Micromorphic probe — prescribe (F̄, v, g) ---------------------------
Fbar_target = np.array([[0.95, 0.0],
                        [0.00, 1.0]])
v_target = np.zeros(N_MODES)
v_target[0] = 0.5
g_target = np.zeros((N_MODES, 2))
g_target[0] = (-0.1, 0.05)

logger.info("Micromorphic solve: F̄_xx=%.3f, v[0]=%.2f, g[0]=%s …",
            Fbar_target[0, 0], v_target[0], np.round(g_target[0], 3))
results = solver(Fbar_target, v_target, g_target)
final = results[-1]

# Constraint violation check (⟨w⟩, ⟨w·φᵢ⟩, ⟨(w·φᵢ)X⟩ should all be ~0).
w_vec = solver.u.x.petsc_vec
c_vecs = solver._newton._constraint_vecs
if c_vecs:
    viols = [abs(c.dot(w_vec)) for c in c_vecs]
    logger.info("Constraint violations (should be ~0): max=%.3e", max(viols))

if comm.rank == 0:
    logger.info("── Effective quantities (final load) ──")
    for key in ("Fbar", "Pbar", "Pi", "Lambda"):
        logger.info("  %-8s shape=%s\n%s", key, np.shape(final[key]), np.asarray(final[key]))
    logger.info("── Tangent block norms ──")
    for key in sorted(k for k in final if k.startswith("d")):
        arr = np.asarray(final[key])
        logger.info("  %-15s shape=%s  |.|=%.4e", key, arr.shape, np.linalg.norm(arr))
    np.savez(os.path.join(output_dir, "micromorphic_results.npz"),
             **{k: np.asarray(v) for k, v in final.items()})
    logger.info("Saved results to %s/micromorphic_results.npz", output_dir)


# --- History plots (mode 0) -------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Fbar_hist = np.array([np.asarray(r["Fbar"]) for r in results])
Pbar_hist = np.array([np.asarray(r["Pbar"]) for r in results])
Pi_hist = np.array([np.asarray(r["Pi"]) for r in results])
Lam_hist = np.array([np.asarray(r["Lambda"]) for r in results])
A_hist = np.array([np.asarray(r["dPbar_dFbar"]) for r in results])
dPidv_h = np.array([np.asarray(r["dPi_dv"]) for r in results])
dLamdg_h = np.array([np.asarray(r["dLambda_dg"]) for r in results])

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
    plot_path = os.path.join(output_dir, "micromorphic_history.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved history plots to %s", plot_path)
