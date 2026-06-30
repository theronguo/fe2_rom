"""
Build the POD + ECM reduced-order model from the full-order snapshots.

Run (serial only):
    python build_rom.py

Reads `output/snapshots/{u_fluc,P}_*.npy` written by run_homogenization.py and
writes the ECM artefacts (basis + magic points) to `ecm/`. The POD basis is
purely geometric (built from the fluctuation / stress snapshots), so the same
artefacts serve the second-order (ch2) reduced solver.
"""
import numpy as np
from dolfinx import io, fem
from mpi4py import MPI
from fe2_rom.rom.pod import POD
from fe2_rom.rom.ecm import ECM

comm = MPI.COMM_WORLD
gdim = 3
degree = 1
snapshot_dir = "output"
ecm_dir = "ecm"
mesh_file = "sphere_strut_2x2x2.msh"
ecm_tol = 1e-8
ratio_uP = 1.0
ratio_P = 1.0
energy_tol = 0.9999999

mesh = io.gmsh.read_from_msh(f"{mesh_file}", comm, 0, gdim=gdim).mesh
V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))

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
    compress_uP="auto",
    quad_degree=degree + 2,   # order-2 (curved) tets: over-integrate the tabulation
)
ecm.compute_magic(tol=ecm_tol)

print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes")
print("Number of magic points:", len(ecm.magic_points))
print(f"Verified full-system residual: {ecm.true_residual:.3e}")

ecm.save_variant2(f"{ecm_dir}")
