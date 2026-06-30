"""
Run:
    python run_homogenization.py
    mpirun -n 4 python run_homogenization.py
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
from fe2_rom.ch2.microsolver import MicroSolver

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

# material parameters (used by the hyperelastic model)
E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

output_dir = "output"
material = NeoHookean(mu=mu, lmbda=lmbda)
solver = MicroSolver(
    mesh_path="sphere_strut_2x2x2.msh",
    comm=comm,
    gdim=3,
    material=material,
    degree=1,
    output_dir=output_dir,
    check_stability=True, 
    visualize_fields=["u_fluc", "u_total", "P"],
    average_quantities=["F", "W", "P"],
    newton_options={"switch_to_minres": True},
    timestepper_options={"t_end": 1.0, "dt_init": 0.05},
    save_snapshots=["u_fluc", "P"],
    rve_volume=8,
    # quadrature_degree=2
)

Fbar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]])

# Second-order macro load: Gbar_iJK = d Fbar_iJ / d X_K, symmetric in (J, K).
# Set to all-zeros to recover the first-order (ch1) response exactly.
Gbar = np.zeros((3, 3, 3))

res = solver(Fbar, Gbar, pert_amplitude_init=1e-3)

Fbar_conv = []
Pbar_conv = []
Wbar_conv = []
for q in res:
    # Only collect quantities that were actually requested via
    # average_quantities — otherwise the keys are absent (KeyError).
    if "Fbar" in q:
        Fbar_conv.append(q["Fbar"])
    if "Pbar" in q:
        Pbar_conv.append(q["Pbar"])
    if "Wbar" in q:
        Wbar_conv.append(q["Wbar"])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)
Wbar_conv = np.array(Wbar_conv)

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
