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
phi_files = sorted(f for f in glob(f"{phi_dir}/phi_*.npy") if "dof_coords" not in f)
phi_snapshots = np.array([np.load(f) for f in phi_files])
N_modes = phi_snapshots.shape[0]
phi_fns = []
for i in range(N_modes):
    fn = fem.Function(V, name=f"phi_{i}")
    fn.x.array[:] = phi_snapshots[i]
    fn.x.scatter_forward()
    phi_fns.append(fn)

# Each Pi_i (scalar) and Lambda_i (gdim-vector) is an independent density —
# coupling to the independent macro variables v_i and g_i — so we POD them
# per-mode rather than stacking the modes into one vector field.
S_Pi_i = fem.functionspace(mesh, ("DG", 1))
S_Lambda_i = fem.functionspace(mesh, ("DG", 1, (gdim,)))

P_func = fem.Function(S, name="P_snap")
X = ufl.SpatialCoordinate(mesh)

pi_exprs = [
    fem.Expression(
        ufl.inner(P_func, ufl.grad(phi)),
        S_Pi_i.element.interpolation_points,
    )
    for phi in phi_fns
]
lam_exprs = []
for phi in phi_fns:
    Pphi = ufl.dot(phi, P_func)
    Pgradphi = ufl.inner(P_func, ufl.grad(phi))
    lam_exprs.append(fem.Expression(
        ufl.as_vector([Pphi[d] + X[d] * Pgradphi for d in range(gdim)]),
        S_Lambda_i.element.interpolation_points,
    ))

pi_fn_i = fem.Function(S_Pi_i)
lam_fn_i = fem.Function(S_Lambda_i)
snapshots_Pi_list = [
    np.zeros((snapshots_P.shape[0], pi_fn_i.x.array.size))
    for _ in range(N_modes)
]
snapshots_Lambda_list = [
    np.zeros((snapshots_P.shape[0], lam_fn_i.x.array.size))
    for _ in range(N_modes)
]
for t in range(snapshots_P.shape[0]):
    P_func.x.array[:] = snapshots_P[t]
    P_func.x.scatter_forward()
    for i in range(N_modes):
        pi_fn_i.interpolate(pi_exprs[i])
        lam_fn_i.interpolate(lam_exprs[i])
        snapshots_Pi_list[i][t] = pi_fn_i.x.array
        snapshots_Lambda_list[i][t] = lam_fn_i.x.array

pod_Pi_list, pod_Lambda_list, N_Pi_list, N_Lambda_list = [], [], [], []
for i in range(N_modes):
    pod_Pi_i = POD(snapshots_Pi_list[i], S_Pi_i, inner_product="L2")
    pod_Lam_i = POD(snapshots_Lambda_list[i], S_Lambda_i, inner_product="L2")
    pod_Pi_i.plot_eigenvalues()
    pod_Pi_list.append(pod_Pi_i)
    pod_Lambda_list.append(pod_Lam_i)
    N_Pi_list.append(pod_Pi_i.n_modes(energy_tol))
    N_Lambda_list.append(pod_Lam_i.n_modes(energy_tol))
    print(f"Mode {i}: Pi_{i} POD: {N_Pi_list[i]} modes;  "
          f"Lambda_{i} POD: {N_Lambda_list[i]} modes")

ecm_kwargs: dict = {}
for i in range(N_modes):
    n_pi = N_Pi_list[i]
    ecm_kwargs[f"Pi_{i}"] = {
        "basis": pod_Pi_list[i].basis[:, :n_pi],
        "space": S_Pi_i,
        "sigma": np.sqrt(pod_Pi_list[i].eigenvalues[:n_pi]),
        "ratio": ratio_Pi,
    }
    n_lam = N_Lambda_list[i]
    ecm_kwargs[f"Lambda_{i}"] = {
        "basis": pod_Lambda_list[i].basis[:, :n_lam],
        "space": S_Lambda_i,
        "sigma": np.sqrt(pod_Lambda_list[i].eigenvalues[:n_lam]),
        "ratio": ratio_Lambda,
    }

ecm = ECM(
    pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
    degree=degree,
    sigma_u=np.sqrt(pod_u.eigenvalues[:N]),
    sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
    ratio_uP=ratio_uP, ratio_P=ratio_P,
    kwargs=ecm_kwargs,
)
ecm.compute_magic(tol=ecm_tol)

print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes, "
      f"N_Pi_per_mode={N_Pi_list}, N_Lambda_per_mode={N_Lambda_list}")
print("Number of magic points:", len(ecm.magic_points))

ecm.save_variant2(f"{ecm_dir}")
