"""Macro micromorphic FE² driver — dummy constitutive law.

Verifies the mixed (u, v_1) formulation on a unit-cube 3D mesh using a
linear, decoupled dummy material (no RVE).  Run with::

    mamba activate fe2_rom_env
    python run_macro_micromorphic.py

Expected: SNES converges every step in a few Newton iterations; P, Pi, Lambda
fields are non-zero; final residual norm < SNES atol.
"""

import os
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, CellType

import logging
from fe2_rom.hyperelastic_solver.logging_utils import setup_logging
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper
from fe2_rom.macro_solver.material_micromorphic import DummyMicromorphicMaterial
from fe2_rom.macro_solver.macro_micromorphic import MacroMicromorphicSolver

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_MODES  = 1
N_ELEM   = 4        # mesh divisions in each direction
N_QP     = 2        # quadrature degree (dolfinx_materials convention: points per dir)
MAX_DISP = 0.05     # total applied displacement in x
N_STEPS  = 5

setup_logging(MPI.COMM_WORLD, level=logging.INFO)

# ---------------------------------------------------------------------------
# Mesh: unit cube, hexahedral cells
# ---------------------------------------------------------------------------
mesh = create_unit_cube(MPI.COMM_WORLD, N_ELEM, N_ELEM, N_ELEM, CellType.hexahedron)

# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
material = DummyMicromorphicMaterial(
    N_modes=N_MODES,
    gdim=3,
    mu=1.0,
    alpha=1.0,
    beta=1.0,
)

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
solver = MacroMicromorphicSolver(
    mesh,
    n_qp=N_QP,
    N_modes=N_MODES,
    material=material,
    degree=1,
)

disp_const = fem.Constant(mesh, 0.0)
zero_const = fem.Constant(mesh, 0.0)

# Fix u_x, u_y, u_z = 0 on left face (x = 0)
solver.add_bc(0, lambda x: np.isclose(x[0], 0.0), zero_const)
solver.add_bc(1, lambda x: np.isclose(x[0], 0.0), zero_const)
solver.add_bc(2, lambda x: np.isclose(x[0], 0.0), zero_const)
# Prescribe u_x on right face (x = 1), measure reaction
solver.add_bc(0, lambda x: np.isclose(x[0], 1.0), disp_const, measure_reaction=True)
# Fix v_1 = 0 on all boundaries (homogeneous Dirichlet for enrichment amplitude)
solver.add_bc((1,), lambda x: np.ones(x.shape[1], dtype=bool), zero_const)

solver.setup()

# ---------------------------------------------------------------------------
# Loadhistory and timestepper
# ---------------------------------------------------------------------------
def loadhistory(t: float) -> None:
    disp_const.value = t * MAX_DISP


timestepper = TimeStepper(
    t_end=1.0,
    dt_init=1.0 / N_STEPS,
    dt_min=1e-6,
)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
output_dir = os.path.join(
    os.path.dirname(__file__), "output_macro_micromorphic_dummy"
)

solver.solve(
    output_dir=output_dir,
    timestepper=timestepper,
    loadhistory=loadhistory,
)

print("Run complete.  Output written to", output_dir)
