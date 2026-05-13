"""FE² macro solver demo — single hex cube under uniaxial compression.

Constitutive response at each macro qp comes from a reduced-order RVE
solver (the ECM ROM built for examples/periodic_solver/example_2). The
problem is wired together by :class:`fe2_rom.macro_solver.MacroSolver`.

Run:
    conda activate fe2_rom_env
    python run_macro.py
"""
import os
# Avoid OpenBLAS/OpenMP oversubscription — many nested RVE solvers run serially.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh as dmesh

from fe2_rom.hyperelastic_solver import (
    NeoHookean,
    TimeStepper,
    ReactionForceLogger,
    setup_logging,
)
from fe2_rom.macro_solver import MacroSolver


# --- Logging ----------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.macro_solver.example_1")
logger.addFilter(lambda r: comm.rank == 0)
for name in (
    "fe2_rom.hyperelastic_solver.solver",
    "fe2_rom.hyperelastic_solver.solvers",
    "fe2_rom.rve_rom.solver",
):
    logging.getLogger(name).setLevel(logging.ERROR)


# --- Macro mesh: single hex on [0,1]^3 --------------------------------------
domain = dmesh.create_unit_cube(
    comm, 2, 2, 4,
    cell_type=dmesh.CellType.hexahedron,
    ghost_mode=dmesh.GhostMode.none,
)


# --- RVE setup (matches legacy run_macro.py) --------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.join(HERE, "..", "..", "periodic_solver", "example_2", "mesh.msh")
ROM_DIR  = os.path.join(HERE, "..", "..", "periodic_solver", "example_2", "ecm")

E_micro, nu_micro = 3000.0, 0.30
mu_micro  = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))


# --- Macro solver -----------------------------------------------------------
solver = MacroSolver(
    mesh=domain,
    full=False,
    n_qp=2,
    rve_mesh_path=RVE_MESH,
    rve_material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
    rom_dir=ROM_DIR,
    gdim=3,
    rve_degree=1,
    rve_output_dir="output",
    rve_visualize_fields=[""],
    rve_average_fields=["P", "A"],
    rve_newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                        "max_iter": 50, "div_rel_tol": 10.0},
    rve_timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-3,
                             "dt_max": 1.0, "good_newton_steps": 5},
    rve_averages_only_final=True,
    degree=1,
    check_stability=True,
)


# --- BCs: clamp z=0 (all components), prescribe uz on z=1, measure z reaction
zero = fem.Constant(domain, 0.0)
disp = fem.Constant(domain, 0.0)
for sub in (0, 1, 2):
    solver.add_bc(sub, lambda x: np.isclose(x[2], 0.0), zero)
solver.add_bc(2, lambda x: np.isclose(x[2], 1.0), disp,
              measure_reaction=True, reaction_direction=(0.0, 0.0, 1.0))

solver.setup()


# --- Load history & time stepper --------------------------------------------
disp_total = -0.25  # 25 % uniaxial compression
def loadhistory(t):
    disp.value = disp_total * t

timestepper = TimeStepper(
    t_end=1.0, dt_init=0.05, dt_min=1e-3, dt_max=0.05, good_newton_steps=5,
)
reaction_logger = ReactionForceLogger()

solver.solve(
    output_dir="output",
    timestepper=timestepper,
    loadhistory=loadhistory,
    output_variables=[solver.u],
    reaction_logger=reaction_logger,
    pert_amplitude_init=0.1
)

output_dir = "output"
reaction_logger.save(
    comm,
    os.path.join(output_dir, "reaction.png"),
    os.path.join(output_dir, "reaction.csv"),
)
