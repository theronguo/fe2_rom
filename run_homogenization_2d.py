"""
Run:
    python run_solver.py
    mpirun -n 4 python run_solver.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # avoid OpenBLAS oversubscription
os.environ["OMP_NUM_THREADS"] = "1"      # avoid OpenMP oversubscription
os.environ["MKL_NUM_THREADS"] = "1"      # avoid MKL oversubscription
import logging
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import io
from mpi4py import MPI
from hyperelastic_solver import (
    PeriodicHyperelasticHomogenizationSolver,
    NeoHookean,
    VTXManager,
    setup_logging,
)

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("holes.msh", comm, 0, gdim=2)

# material parameters (used by the hyperelastic model)
E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

material = NeoHookean(mu=mu, lmbda=lmbda)
solver = PeriodicHyperelasticHomogenizationSolver(
    mesh, cell_tags, facet_tags, material,
    enable_viz_fields=True, degree=2
)
solver.setup(check_stability=True, 
             newton_options={"switch_to_minres": True},
             timestepper_options={"t_end": 1.0, "dt_init": 1.0})

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
vtx = VTXManager(comm, f"{output_dir}/solution.bp",
                 [solver.u_int, solver.F_func, solver.P_func, solver.J_func, solver.u_total])
res = solver(np.array([[0.8, 0.0], [0.0, 1.0]]),
    pert_amplitude_init=1e-1,
    output_manager=vtx,
    return_quantities=["F", "P"],
)
res = solver(np.array([[0.8, 0.0], [0.0, 0.8]]),
    pert_amplitude_init=1e-1,
    output_manager=vtx, plot_time_start=1.0,
    return_quantities=["F", "P"],
)
vtx.close()

Fbar_conv = []
Pbar_conv = []
for q in res:
    Fbar_conv.append(q[0])
    Pbar_conv.append(q[1])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)

if comm.rank == 0 and Fbar_conv.size and Pbar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 1, 1], Pbar_conv[:, 1, 1], marker="o")
    ax.set_xlabel("Fyy")
    ax.set_ylabel("Pyy")
    ax.set_title("Pyy over Fyy")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Pyy_over_Fyy.pdf", dpi=300)
    plt.close(fig)

