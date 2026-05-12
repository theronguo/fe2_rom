"""
Run:
    python build_rom.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from dolfinx import io, fem
from mpi4py import MPI
from fe2_rom.rve_rom.pod import POD, ECM

comm = MPI.COMM_WORLD
gdim = 2
degree = 2
snapshot_dir = "output"
ecm_dir = "ecm"
mesh_file = "rve.msh"
ecm_tol = 1e-6
ratio_uP = 10.0
ratio_P = 2.0
energy_tol = 0.9999

mesh, _, _ = io.gmshio.read_from_msh(f"{mesh_file}", comm, 0, gdim=gdim)
V  = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S  = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))

snapshots_u = POD.load_and_align_snapshots(f"{snapshot_dir}/snapshots/u_fluc_*.npy", V)
pod_u = POD(snapshots_u, V, inner_product="H1")
N = pod_u.n_modes(energy_tol)

snapshots_P = POD.load_and_align_snapshots(f"{snapshot_dir}/snapshots/P_*.npy", S)
pod_P = POD(snapshots_P, S, inner_product="L2")
M = pod_P.n_modes(energy_tol)

ecm = ECM(
    pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
    degree=degree,
    sigma_u=np.sqrt(pod_u.eigenvalues[:N]),
    sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
    ratio_uP=ratio_uP, ratio_P=ratio_P,
)
ecm.compute_magic(tol=ecm_tol)

print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes")
print("Number of magic points:", len(ecm.magic_points))

ecm.save_variant2(f"{ecm_dir}")
