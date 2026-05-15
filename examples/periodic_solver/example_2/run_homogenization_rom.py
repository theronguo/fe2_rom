"""
Run:
    python run_homogenization_rom.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # avoid OpenBLAS oversubscription
os.environ["OMP_NUM_THREADS"] = "1"      # avoid OpenMP oversubscription
os.environ["MKL_NUM_THREADS"] = "1"      # avoid MKL oversubscription
import logging
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.rve_rom.solver import RVESolver

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

output_dir = "output_rom"
material = NeoHookean(mu=mu, lmbda=lmbda)
solver = RVESolver(
    mesh_path="mesh.msh",
    rom_dir="ecm",
    material=material,
    comm=comm,
    gdim=3,
    degree=1,
    output_dir=output_dir,
    visualize_fields=[""],
    average_quantities=["F", "P", "A"],
    timestepper_options={"t_end": 1.0, "dt_init": 0.01, "dt_min": 1e-5, "dt_max": 0.01, "good_newton_steps": 5},
)

res = solver(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.75]]))

Fbar_conv = []
Pbar_conv = []
Abar_conv = []
for q in res:
    Fbar_conv.append(q["Fbar"])
    Pbar_conv.append(q["Pbar"])
    Abar_conv.append(q["dPbar_dFbar"])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)
Abar_conv = np.array(Abar_conv)

os.makedirs(output_dir, exist_ok=True)
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
