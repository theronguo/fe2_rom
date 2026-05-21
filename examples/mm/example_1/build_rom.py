"""
Run:
    python build_rom.py
    mpirun -n 4 python build_rom.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from glob import glob

import numpy as np
import ufl
from dolfinx import io, fem
from mpi4py import MPI
from scipy.spatial import cKDTree

from fe2_rom.rom.pod import POD
from fe2_rom.rom.ecm import ECM

comm = MPI.COMM_WORLD
gdim = 2
degree = 2
snapshot_dir = "output"
pool_dir = f"{snapshot_dir}/snapshots_pool"
phi_dir = f"{snapshot_dir}/snapshots"
ecm_dir = "ecm"
mesh_file = "rve.msh"
ecm_tol = 1e-4
ratio_uP = 1.0
ratio_P = 1e0
ratio_Pi = 1e0
ratio_Lambda = 1e0
energy_tol = 0.999999

mesh = io.gmsh.read_from_msh(f"{mesh_file}", comm, 0, gdim=gdim).mesh
V  = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S  = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))


def load_pool_snapshots(field: str, V_space):
    """Load all ``{field}_s*_*.npy`` snapshots from the merged pool and align
    DOFs to ``V_space``'s serial ordering using the shared
    ``{field}_dof_coords.npy`` written by the sampler.
    """
    files = sorted(
        f for f in glob(f"{pool_dir}/{field}_s*_*.npy")
        if "dof_coords" not in f
    )
    if not files:
        raise FileNotFoundError(f"No snapshots for field '{field}' in {pool_dir}")
    snaps = np.array([np.load(f) for f in files])

    coords_path = f"{pool_dir}/{field}_dof_coords.npy"
    try:
        saved_coords = np.load(coords_path)
    except FileNotFoundError:
        return snaps

    serial_coords = V_space.tabulate_dof_coordinates()
    bs = V_space.dofmap.index_map_bs
    _, perm = cKDTree(saved_coords).query(serial_coords, k=1)
    dof_perm = (perm[:, None] * bs + np.arange(bs)).ravel()
    return snaps[:, dof_perm]


snapshots_u = load_pool_snapshots("u_fluc", V)
print(f"[snapshots] u_fluc: {snapshots_u.shape}")
pod_u = POD(snapshots_u, V, inner_product="L2")
N = pod_u.n_modes(energy_tol)

u_proj = snapshots_u @ pod_u._ip_matrix @ pod_u.basis[:, :N] @ pod_u.basis[:, :N].T
h1_err = np.sqrt(np.diagonal((snapshots_u - u_proj) @ pod_u._ip_matrix @ (snapshots_u - u_proj).T))
h1_norm = np.sqrt(np.diagonal(snapshots_u @ pod_u._ip_matrix @ snapshots_u.T))
reconstruction_error = h1_err / h1_norm
print(f"Max reconstruction error among snapshots: {reconstruction_error.max():.2%}, mean: {reconstruction_error.mean():.2%}")

snapshots_P = load_pool_snapshots("P", S)
print(f"[snapshots] P:      {snapshots_P.shape}")
pod_P = POD(snapshots_P, S, inner_product="L2")
M = pod_P.n_modes(energy_tol)

P_proj = snapshots_P @ pod_P._ip_matrix @ pod_P.basis[:, :M] @ pod_P.basis[:, :M].T
l2_err = np.sqrt(np.diagonal((snapshots_P - P_proj) @ pod_P._ip_matrix @ (snapshots_P - P_proj).T))
l2_norm = np.sqrt(np.diagonal(snapshots_P @ pod_P._ip_matrix @ snapshots_P.T))
reconstruction_error = l2_err / l2_norm
print(f"Max reconstruction error among snapshots: {reconstruction_error.max():.2%}, mean: {reconstruction_error.mean():.2%}")

# --- Pi and Lambda density snapshots, built from phi modes ----------------
# phi modes were saved by compute_linear_buckling_modes(save_modes=True) as
# phi_<i>.npy in the same snapshots/ folder; they live on V.
phi_snapshots = POD.load_and_align_snapshots(
    f"{phi_dir}/phi_*.npy", V,
)
N_modes = phi_snapshots.shape[0]
phi_fns = []
for i in range(N_modes):
    fn = fem.Function(V, name=f"phi_{i}")
    fn.x.array[:] = phi_snapshots[i]
    fn.x.scatter_forward()
    phi_fns.append(fn)

# Density spaces: stack across modes (and gdim for Lambda) so each timestep
# yields a single field per quantity.
S_Pi = fem.functionspace(mesh, ("DG", 1, (N_modes,)))
S_Lambda = fem.functionspace(mesh, ("DG", 1, (N_modes, gdim)))

P_func = fem.Function(S, name="P_snap")
X = ufl.SpatialCoordinate(mesh)
pi_components = [ufl.inner(P_func, ufl.grad(phi)) for phi in phi_fns]
pi_expr = fem.Expression(
    ufl.as_vector(pi_components),
    S_Pi.element.interpolation_points,
)
lam_rows = []
for phi in phi_fns:
    Pphi = ufl.dot(phi, P_func)            # vector of length gdim
    Pgradphi = ufl.inner(P_func, ufl.grad(phi))
    lam_rows.append(
        ufl.as_vector([Pphi[d] + X[d] * Pgradphi for d in range(gdim)])
    )
lam_expr = fem.Expression(
    ufl.as_matrix([[row[d] for d in range(gdim)] for row in lam_rows]),
    S_Lambda.element.interpolation_points,
)

pi_fn = fem.Function(S_Pi)
lam_fn = fem.Function(S_Lambda)
snapshots_Pi = np.zeros((snapshots_P.shape[0], pi_fn.x.array.size))
snapshots_Lambda = np.zeros((snapshots_P.shape[0], lam_fn.x.array.size))
for t in range(snapshots_P.shape[0]):
    P_func.x.array[:] = snapshots_P[t]
    P_func.x.scatter_forward()
    pi_fn.interpolate(pi_expr)
    lam_fn.interpolate(lam_expr)
    snapshots_Pi[t] = pi_fn.x.array
    snapshots_Lambda[t] = lam_fn.x.array

pod_Pi = POD(snapshots_Pi, S_Pi, inner_product="L2")
pod_Lambda = POD(snapshots_Lambda, S_Lambda, inner_product="L2")
N_Pi = pod_Pi.n_modes(energy_tol)
N_Lambda = pod_Lambda.n_modes(energy_tol)
print(f"Pi POD: {N_Pi} modes;  Lambda POD: {N_Lambda} modes")
ecm = ECM(
    pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
    degree=degree,
    sigma_u=np.sqrt(pod_u.eigenvalues[:N]),
    sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
    ratio_uP=ratio_uP, ratio_P=ratio_P,
    kwargs={"Pi": {"basis": pod_Pi.basis[:, :N_Pi], "space": S_Pi, "sigma": np.sqrt(pod_Pi.eigenvalues[:N_Pi]), "ratio": ratio_Pi},
            "Lambda": {"basis": pod_Lambda.basis[:, :N_Lambda], "space": S_Lambda, "sigma": np.sqrt(pod_Lambda.eigenvalues[:N_Lambda]), "ratio": ratio_Lambda}},
)
ecm.compute_magic(tol=ecm_tol)

print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes, N_Pi={N_Pi} Pi-modes, N_Lambda={N_Lambda} Lambda-modes")
print("Number of magic points:", len(ecm.magic_points))

ecm.save_variant2(f"{ecm_dir}")
