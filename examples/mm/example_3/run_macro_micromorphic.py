"""Macro micromorphic FE² driver — real RVE constitutive law.

Uses ``MicromorphicRVEMaterial`` backed by either:

* ``ReducedMicroSolver`` (ROM, default),
* ``MicroSolver`` (FOM, set USE_ROM=False).

Prerequisites (ROM):
    1. In ``../example_1/``, run ``run_micromorphic.py`` with ``save_modes=True``
       — produces ``../example_1/output/snapshots/phi_*.npy``.
    2. In ``../example_1/``, run ``build_rom.py`` — produces ``../example_1/ecm/``.

Prerequisites (FOM):
    Same step 1 only (phi modes must exist in ``../example_1/output/snapshots/``).

Run (ROM)::

    mamba activate fe2_rom_env
    python run_macro_micromorphic_rve.py

Run (FOM, single-threaded to avoid oversubscription)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_macro_micromorphic_rve.py

Expected: SNES converges every step; Newton iteration count ≤ 25; P, Pi,
Lambda fields are non-zero; reaction force grows with applied displacement.
Quadratic Newton convergence (2–4 iterations) validates the tangent blocks.
"""

import os
# Keep nested RVE solves single-threaded to avoid oversubscription.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import glob
import logging
import re

import numpy as np
from mpi4py import MPI

from dolfinx.mesh import create_unit_square, CellType
from dolfinx import fem, mesh as dmesh

from fe2_rom.hyperelastic_solver import NeoHookean, ReactionForceLogger, setup_logging, broadcast_logger
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper
from fe2_rom.mm.material import MicromorphicRVEMaterial
from fe2_rom.mm.macrosolver import MacroMicromorphicSolver

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
USE_ROM    = False     # True → ReducedMicroSolver, False → FOM homogenization
N_ELEM     = 4        # macro mesh divisions per direction
N_QP       = 2        # quadrature degree for dolfinx_materials
MAX_DISP   = -0.02     # total applied u_x displacement (keep small for linear regime)
N_STEPS    = 4        # number of macro load steps

# RVE micro material parameters
E_MICRO, NU_MICRO = 3000.0, 0.30
MU_MICRO  = E_MICRO / (2.0 * (1.0 + NU_MICRO))
LAM_MICRO = E_MICRO * NU_MICRO / ((1.0 + NU_MICRO) * (1.0 - 2.0 * NU_MICRO))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
EXAMPLE1 = os.path.join(HERE, "..", "example_1")
RVE_MESH = os.path.join(EXAMPLE1, "rve.msh")
ROM_DIR  = os.path.join(EXAMPLE1, "ecm")
PHI_DIR  = os.path.join(EXAMPLE1, "output", "snapshots")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.macro_solver.example_2")
logger.addFilter(lambda _: comm.rank == 0)

VERBOSE_RVE = False  # set True to see per-qp Newton/stability output
_RVE_LOGGERS = (
    "fe2_rom.hyperelastic_solver.solver",
    "fe2_rom.hyperelastic_solver.solvers",
    "fe2_rom.hyperelastic_solver.stability",
    "fe2_rom.ch1.microsolver",
    "fe2_rom.rom.solver_ch1"
)
if VERBOSE_RVE:
    broadcast_logger(*_RVE_LOGGERS, level=logging.DEBUG)
else:
    for name in _RVE_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Load φ-mode arrays (produced by run_micromorphic.py with save_modes=True)
# ---------------------------------------------------------------------------
phi_files = sorted(
    f for f in glob.glob(os.path.join(PHI_DIR, "phi_*.npy"))
    if re.search(r"phi_[\d.]+\.npy$", os.path.basename(f))
)
if not phi_files:
    raise FileNotFoundError(
        f"No phi_*.npy found in {PHI_DIR}.\n"
        "Run 'python run_micromorphic.py' first with save_modes=True."
    )
phi_arrays = [np.load(fp) for fp in phi_files]
N_MODES = len(phi_arrays)
logger.info("Loaded %d φ mode(s) from %s", N_MODES, PHI_DIR)

# ---------------------------------------------------------------------------
# RVE factory — one RVE instance per macro quadrature point
# ---------------------------------------------------------------------------
if USE_ROM:
    from fe2_rom.rom import ReducedMicroSolver

    def rve_factory(rank: int, index: int):
        out_dir = os.path.join(HERE, "output_macro_rve", f"rve_{rank}_{index}")
        return ReducedMicroSolver(
            mesh_path=RVE_MESH,
            rom_dir=ROM_DIR,
            material=NeoHookean(mu=MU_MICRO, lmbda=LAM_MICRO),
            phi=phi_arrays,
            gdim=2,
            degree=2,
            comm=MPI.COMM_SELF,
            output_dir=out_dir,
            visualize_fields=[],
            newton_options={
                "rel_tol": 1e-8, "abs_tol": 1e-6,
                "max_iter": 50, "div_rel_tol": 10.0,
            },
            timestepper_options={
                "t_end": 1.0, "dt_init": 0.1, "dt_min": 1e-5,
                "dt_max": 0.5, "good_newton_steps": 5,
            },
            averages_only_final=True,
        )

    logger.info("Using ROM RVE solver (ecm dir: %s)", ROM_DIR)

else:
    from fe2_rom.mm.microsolver import MicroSolver

    def rve_factory(rank: int, index: int):
        out_dir = os.path.join(HERE, "output_macro_rve", f"rve_{rank}_{index}")
        rve = MicroSolver(
            mesh_path=RVE_MESH,
            comm=MPI.COMM_SELF,
            gdim=2,
            material=NeoHookean(mu=MU_MICRO, lmbda=LAM_MICRO),
            N=N_MODES,
            degree=2,
            output_dir=out_dir,
            visualize_fields=[],
            newton_options={
                "rel_tol": 1e-8, "abs_tol": 1e-6,
                "max_iter": 50, "div_rel_tol": 10.0,
                "switch_to_minres": True,
            },
            timestepper_options={
                "t_end": 1.0, "dt_init": 0.1, "dt_min": 1e-5,
                "dt_max": 0.5, "good_newton_steps": 5,
            },
            averages_only_final=True,
        )
        # Populate φ from pre-computed arrays (saved by run_micromorphic.py).
        # The solver was run serially, so DOF order matches COMM_SELF here.
        for i, arr in enumerate(phi_arrays):
            rve._phi[i].x.array[:] = arr
            rve._phi[i].x.scatter_forward()
        rve.rebuild_constraints()
        return rve

    logger.info("Using FOM RVE solver (mesh: %s)", RVE_MESH)

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
material = MicromorphicRVEMaterial(rve_factory, N_modes=N_MODES, gdim=2)

# ---------------------------------------------------------------------------
# Macro mesh and solver
# ---------------------------------------------------------------------------
mesh = create_unit_square(comm, N_ELEM, N_ELEM, CellType.quadrilateral, ghost_mode=dmesh.GhostMode.none)

solver = MacroMicromorphicSolver(
    mesh,
    n_qp=N_QP,
    N_modes=N_MODES,
    material=material,
    degree=1,
)

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
disp_const = fem.Constant(mesh, 0.0)
zero_const = fem.Constant(mesh, 0.0)

# Fix u_x and u_y on left edge (x = 0)
solver.add_bc((0, 0), lambda x: np.isclose(x[0], 0.0), zero_const)
solver.add_bc((0, 1), lambda x: np.isclose(x[0], 0.0), zero_const)
# Prescribe u_x on right edge (x = 1) and measure reaction
solver.add_bc(
    (0, 0),
    lambda x: np.isclose(x[0], 1.0),
    disp_const,
    measure_reaction=True,
)
# Homogeneous Dirichlet for enrichment amplitudes on all boundaries
for i in range(N_MODES):
    solver.add_bc((i + 1,), lambda x: np.ones(x.shape[1], dtype=bool), zero_const)

solver.setup()

# ---------------------------------------------------------------------------
# Load history and timestepper
# ---------------------------------------------------------------------------
def loadhistory(t: float) -> None:
    disp_const.value = t * MAX_DISP


timestepper = TimeStepper(
    t_end=1.0,
    dt_init=1.0 / N_STEPS,
    dt_min=1e-6,
    dt_max=1.0 / N_STEPS,
)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
mode_label = "rom" if USE_ROM else "fom"
output_dir = os.path.join(HERE, f"output_{mode_label}")

reaction_logger = ReactionForceLogger()

solver.solve(
    output_dir=output_dir,
    timestepper=timestepper,
    loadhistory=loadhistory,
    reaction_logger=reaction_logger,
)

reaction_logger.save(
    comm,
    os.path.join(output_dir, "reaction.png"),
    os.path.join(output_dir, "reaction.csv"),
)

logger.info("Run complete.  Output written to %s", output_dir)
