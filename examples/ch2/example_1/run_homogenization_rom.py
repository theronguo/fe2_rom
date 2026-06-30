"""
Second-order (CH2) reduced-order online homogenization.

Run (serial only):
    python run_homogenization_rom.py

Drives the POD + ECM reduced RVE (built by build_rom.py, read from `ecm/`) with
the same macroscopic load as the full-order run and writes the reduced response
to `output_rom/`. Called as solver(Fbar, Gbar); a zero Gbar recovers the
first-order (ch1) response.
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
from fe2_rom.rom.solver_ch2 import ReducedMicroSolver

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)

# material parameters (used by the hyperelastic model)
E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

output_dir = "output_rom"
material = NeoHookean(mu=mu, lmbda=lmbda)
solver = ReducedMicroSolver(
    mesh_path="sphere_strut_2x2x2.msh",
    rom_dir="ecm",
    material=material,
    comm=comm,
    gdim=3,
    degree=1,
    output_dir=output_dir,
    visualize_fields=["u_fluc", "u_total"],
    average_quantities=["F", "W", "P"],
    timestepper_options={"t_end": 1.0, "dt_init": 0.05, "dt_min": 1e-5, "dt_max": 0.05, "good_newton_steps": 5},
    rve_volume=8,
)

Fbar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.8]])

# Second-order macro load: Gbar_iJK = d Fbar_iJ / d X_K, symmetric in (J, K).
# Set to all-zeros to recover the first-order (ch1) response exactly.
Gbar = np.zeros((3, 3, 3))

res = solver(Fbar, Gbar)

Fbar_conv = []
Pbar_conv = []
Wbar_conv = []
for q in res:
    if "Fbar" in q:
        Fbar_conv.append(q["Fbar"])
    if "Pbar" in q:
        Pbar_conv.append(q["Pbar"])
    if "Wbar" in q:
        Wbar_conv.append(q["Wbar"])
Fbar_conv = np.array(Fbar_conv)
Pbar_conv = np.array(Pbar_conv)
Wbar_conv = np.array(Wbar_conv)

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
