"""
Run:
    python run_solver.py
    mpirun -n 4 python run_solver.py
"""
import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # avoid OpenBLAS oversubscription
os.environ["OMP_NUM_THREADS"] = "1"      # avoid OpenMP oversubscription
os.environ["MKL_NUM_THREADS"] = "1"      # avoid MKL oversubscription
import numpy as np
from dolfinx import fem, io
from mpi4py import MPI
from petsc4py import PETSc

from fe2_rom.hyperelastic_solver import (
    HyperelasticStabilitySolver,
    NeoHookean,
    ReactionForceLogger,
    TimeStepper,
    VTXManager,
    setup_logging,
)

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("hex_lattice.msh", comm, 0, gdim=3)

material = NeoHookean(mu=1000.0, lmbda=2000.0)
solver = HyperelasticStabilitySolver(
    mesh, cell_tags, facet_tags, material,
    enable_viz_fields=True,
)

const_0 = fem.Constant(mesh, PETSc.ScalarType(0.0))
const_1 = fem.Constant(mesh, PETSc.ScalarType(0.0))

# z_min: fully clamped
solver.add_bc(0, lambda x: x[2] < 0+1e-8, const_0)
solver.add_bc(1, lambda x: x[2] < 0+1e-8, const_0)
solver.add_bc(2, lambda x: x[2] < 0+1e-8, const_0)

# z_max: lateral fixed, prescribed vertical displacement + reaction force probe
solver.add_bc(2, lambda x: x[2] > 1-1e-8, const_1,
              measure_reaction=True, reaction_direction=(0.0, 0.0, 1.0))

solver.setup(check_stability=True, newton_options={"switch_to_minres": True})

max_amplitude = -0.5
def load_schedule(t: float) -> None:
    const_1.value = t * max_amplitude


os.makedirs("output", exist_ok=True)
vtx = VTXManager(comm, "output/solution.bp",
                 [solver.u_int, solver.F_func, solver.P_func, solver.J_func])
rf_logger = ReactionForceLogger()

solver.run(
    load_schedule,
    timestepper=TimeStepper(
        t_end=1.0,
        dt_init=0.1,
        dt_min=1e-5,
        dt_max=0.1,
        good_newton_steps=7,
    ),
    output_manager=vtx,
    reaction_logger=rf_logger,
    pert_amplitude_init=1e1,
)

# from fe2_rom.hyperelastic_solver.solvers import CylindricalArcLength
# arc = CylindricalArcLength(
#     arc_length=15,        # arc-length step in (U, λ) space; increase for larger λ steps
#     max_arc_steps=800,
#     max_newton_iter=15,
#     abs_tol=1e-5,
# )

# solver.run_arc_length(
#     arc,
#     load_fn=load_schedule,
#     lambda_init=0.0,
#     lambda_max=1.0,
#     output_manager=vtx,
#     reaction_logger=rf_logger,
# )


vtx.close()
rf_logger.save(comm, "output/reaction_force.png", "output/reaction_force.csv")
