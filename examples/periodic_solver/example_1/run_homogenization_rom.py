"""
Run:
    python run_homogenization_2d_rom.py
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
    mesh_path="rve.msh",
    rom_dir="ecm",
    material=material,
    comm=comm,
    gdim=2,
    degree=2,
    output_dir=output_dir,
    visualize_fields=[""],
    average_fields=["F", "P", "A"],
    timestepper_options={"t_end": 1.0, "dt_init": 0.01, "dt_min": 1e-5, "dt_max": 0.01, "good_newton_steps": 5},
)

res = solver(np.array([[0.8, 0.0], [0.0, 1.0]]))

Fbar_conv = []
Pbar_conv = []
Abar_conv = []
for q in res:
    Fbar_conv.append(q[0])
    Pbar_conv.append(q[1])
    Abar_conv.append(q[2])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)
Abar_conv = np.array(Abar_conv)

os.makedirs(output_dir, exist_ok=True)
if comm.rank == 0 and Fbar_conv.size and Pbar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 0, 0], Pbar_conv[:, 0, 0], marker="o")
    ax.set_xlabel("Fxx")
    ax.set_ylabel("Pxx")
    ax.set_title("Pxx over Fxx")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Pxx_over_Fxx.pdf", dpi=300)
    plt.close(fig)

if comm.rank == 0 and Fbar_conv.size and Abar_conv.size:
    fig, ax = plt.subplots()
    ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 0, 0], marker="o", label="Axxxx")
    ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 1, 1], marker="o", label="Ayyyy")
    ax.plot(Fbar_conv[:, 0, 0], Abar_conv[:, 0, 0, 0, 1], marker="o", label="Axxxy")
    ax.set_xlabel("Fxx")
    ax.set_ylabel("Axxxx")
    ax.set_title("Axxxx over Fxx")
    ax.grid(True)
    plt.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/Axxxx_over_Fxx.pdf", dpi=300)
    plt.close(fig)
