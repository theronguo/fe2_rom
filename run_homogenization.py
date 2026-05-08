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

mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("mesh.msh", comm, 0, gdim=3)

# material parameters (used by the hyperelastic model)
E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

material = NeoHookean(mu=mu, lmbda=lmbda)
solver = PeriodicHyperelasticHomogenizationSolver(
    mesh, cell_tags, facet_tags, material,
    enable_viz_fields=True,
)
solver.setup(check_stability=True, newton_options={"switch_to_minres": True})

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
vtx = VTXManager(comm, f"{output_dir}/solution.bp",
                 [solver.u_int, solver.F_func, solver.P_func, solver.J_func, solver.W_func, solver.u_total])

res = solver(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.75]]),
    output_manager=vtx,
    pert_amplitude_init=1e1,
    return_quantities=["F", "P", "W", "A"],
)
vtx.close()

Fbar_conv = []
Pbar_conv = []
Wbar_conv = []
Abar_conv = []
for q in res:
    Fbar_conv.append(q[0])
    Pbar_conv.append(q[1])
    Wbar_conv.append(q[2])
    Abar_conv.append(q[3])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)
Wbar_conv = np.array(Wbar_conv)
Abar_conv = np.array(Abar_conv)

if comm.rank == 0 and Fbar_conv.size and Pbar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 2, 2], Pbar_conv[:, 2, 2], marker="o")
    ax.set_xlabel("Fzz")
    ax.set_ylabel("Pzz")
    ax.set_title("Pzz over Fzz")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Pzz_over_Fzz.pdf", dpi=300)
    plt.close(fig)

if comm.rank == 0 and Fbar_conv.size and Wbar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 2, 2], Wbar_conv, marker="o")
    ax.set_xlabel("Fzz")
    ax.set_ylabel("Wzz")
    ax.set_title("Wzz over Fzz")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Wzz_over_Fzz.pdf", dpi=300)
    plt.close(fig)

if comm.rank == 0 and Fbar_conv.size and Abar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 2, 2], Abar_conv[:, 2, 2, 2, 2], marker="o")
    ax.set_xlabel("Fzz")
    ax.set_ylabel("Azzzz")
    ax.set_title("Azzzz over Fzz")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Azzzz_over_Fzz.pdf", dpi=300)
    plt.close(fig)