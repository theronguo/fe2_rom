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

from hyperelastic_solver import (
    PeriodicHyperelasticHomogenizationSolver,
    NeoHookean,
    ReactionForceLogger,
    TimeStepper,
    VTXManager,
    setup_logging,
)

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("mesh.msh", comm, 0, gdim=3)

material = NeoHookean(mu=1000.0, lmbda=2000.0)
solver = PeriodicHyperelasticHomogenizationSolver(
    mesh, cell_tags, facet_tags, material,
    enable_viz_fields=True,
)

const_0 = fem.Constant(mesh, PETSc.ScalarType(0.0))
Fbar = solver.F_bar

solver.add_bc(0, lambda x: x[2] < 0+1e-8, const_0)
solver.add_bc(1, lambda x: x[2] < 0+1e-8, const_0)
solver.add_bc(2, lambda x: x[2] < 0+1e-8, const_0)
solver.add_bc(0, lambda x: x[2] > 1-1e-8, const_0)
solver.add_bc(1, lambda x: x[2] > 1-1e-8, const_0)
solver.add_bc(2, lambda x: x[2] > 1-1e-8, const_0)

solver.add_bc(0, lambda x: x[1] < -0.6+1e-8, const_0)
solver.add_bc(1, lambda x: x[1] < -0.6+1e-8, const_0)
solver.add_bc(2, lambda x: x[1] < -0.6+1e-8, const_0)
solver.add_bc(0, lambda x: x[1] > 0.6-1e-8, const_0)
solver.add_bc(1, lambda x: x[1] > 0.6-1e-8, const_0)
solver.add_bc(2, lambda x: x[1] > 0.6-1e-8, const_0)

solver.add_bc(0, lambda x: x[0] < -0.6+1e-8, const_0)
solver.add_bc(1, lambda x: x[0] < -0.6+1e-8, const_0)
solver.add_bc(2, lambda x: x[0] < -0.6+1e-8, const_0)
solver.add_bc(0, lambda x: x[0] > 0.6-1e-8, const_0)
solver.add_bc(1, lambda x: x[0] > 0.6-1e-8, const_0)
solver.add_bc(2, lambda x: x[0] > 0.6-1e-8, const_0)

solver.setup(check_stability=True, newton_options={"switch_to_minres": True})

# Compute max_amplitude using MPI-reduced z extents
z_local = mesh.geometry.x[:, 2]
z_min = comm.allreduce(float(np.min(z_local)), op=MPI.MIN)
z_max = comm.allreduce(float(np.max(z_local)), op=MPI.MAX)
max_amplitude = -0.25

def load_schedule(t: float) -> None:
    Fbar.value[2, 2] = 1 + t * max_amplitude


os.makedirs("output", exist_ok=True)
vtx = VTXManager(comm, "output/solution.bp",
                 [solver.u_int, solver.F_func, solver.P_func, solver.J_func, solver.u_total])

solver.run(
    load_schedule,
    timestepper=TimeStepper(
        t_end=1.0,
        dt_init=0.2,
        dt_min=1e-5,
        dt_max=0.1,
        good_newton_steps=7,
    ),
    output_manager=vtx,
    pert_amplitude_init=1e1,
)

vtx.close()
