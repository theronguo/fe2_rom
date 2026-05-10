from dolfinx import io, fem
from dolfinx.fem import petsc
from mpi4py import MPI
from glob import glob
import numpy as np
import ufl
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
gdim = 2
degree = 2
mesh, _, _ = io.gmshio.read_from_msh("holes.msh", comm, 0, gdim=gdim)

V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))
S0 = fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.dx(domain=mesh)
H1mat = fem.petsc.assemble_matrix(fem.form(ufl.inner(u, v) * dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx))
H1mat.assemble()

u = ufl.TrialFunction(S)
v = ufl.TestFunction(S)
L2mat = fem.petsc.assemble_matrix(fem.form(ufl.inner(u, v) * dx))
L2mat.assemble()

# 
N = 7
M = 7

# POD
def POD(snapshots, H1mat):
    H1_dense = H1mat.convert("dense")
    H1_np = H1_dense.getDenseArray()
    C = snapshots @ H1_np @ snapshots.T

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    basis = snapshots.T @ eigenvectors
    return basis, eigenvalues

def plot_eigenvalues(eigenvalues):
    plt.semilogy(eigenvalues, marker="o")
    plt.xlabel("Mode")
    plt.ylabel("Eigenvalue")
    plt.title("POD Modes")
    plt.show()

def load_snapshots(pattern):
    snapshot_files = glob(pattern)
    snapshots = []
    for file in snapshot_files:
        snapshots.append(np.load(file))
    return np.array(snapshots)

def visualize_mode(func_pt, N, filename, visualization_space, interpolation_space=None):
    fn = fem.Function(visualization_space)
    if interpolation_space is not None:
        fn_int = fem.Function(interpolation_space)
    with io.XDMFFile(comm, filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for i in range(N):
            fn.x.array[:] = func_pt[:, i]
            if interpolation_space is not None:
                fn_int.interpolate(fn)
            else:
                fn_int = fn
            xdmf.write_function(fn_int, i)

def my_ecm(A, b, tol=1e-4):
    magic_points = []
    candidates = np.array(range(dofs_Q0)).tolist()
    alpha = []
    r = b
    A_norm = A / np.linalg.norm(A, axis=0)
    k = 0
    counter_nnls = 0
    print((k, len(alpha), counter_nnls), np.linalg.norm(r) / np.linalg.norm(b))
    while (np.linalg.norm(r) / np.linalg.norm(b)) > tol:
        new_point_idx = np.argmax(A_norm[:, candidates].T @ r)
        magic_points.append(candidates[new_point_idx])
        candidates.remove(candidates[new_point_idx])
        A_current = A[:, magic_points]
        alpha, _, _, _ = lstsq(A_current, b)
        if np.any(alpha < 0):
            alpha = nnls(A[:, magic_points], b)[0]
            counter_nnls += 1
            zero_weights = np.where(alpha < 1e-8)[0]
            delete_idx_mp = []
            for idx in zero_weights:
                delete_idx_mp.append(magic_points[idx])
                candidates.append(magic_points[idx])
            for idx in delete_idx_mp:
                magic_points.remove(idx)
        A_current = A[:, magic_points]
        alpha, _, _, _ = lstsq(A_current, b)
        r = b - A_current @ alpha
        k = k + 1
        print((k, len(alpha), counter_nnls), np.linalg.norm(r) / np.linalg.norm(b))
    # re-normalize weights
    magic_weights = alpha
    magic_points = magic_points

    return magic_points, magic_weights

# Displacement snapshots
snapshot_files = glob("output/snapshots/u_fluc_*.npy")
snapshots = load_snapshots("output/snapshots/u_fluc_*.npy")
basis_u, eigenvalues = POD(snapshots, H1mat)
# plot_eigenvalues(eigenvalues)
# visualize_mode(basis_u, N, f"pod_mode_u.xdmf", V)

# Stresses
snapshot_files = glob("output/snapshots/P_*.npy")
snapshots = load_snapshots("output/snapshots/P_*.npy")
basis_P, eigenvalues = POD(snapshots, L2mat)
# plot_eigenvalues(eigenvalues)
# visualize_mode(basis_P, M, f"pod_mode_P.xdmf", S, S0)


# Empirical quadrature procedure
basis_u = basis_u[:, :N]
basis_P = basis_P[:, :M]

V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))
Q0 = fem.functionspace(mesh, ("DG", 0 ))

# weights contains the volume of each cell
dofs_Q0 = Q0.dofmap.index_map.size_global * Q0.dofmap.index_map_bs
cell_avg = ufl.TestFunction(Q0)
weights = np.zeros(dofs_Q0)
weights = fem.petsc.assemble_vector(fem.form(cell_avg * dx)).array.copy()
volume = weights.sum()

# integrate and pointwise evaluate all pairwise combination of basis_P : grad(basis_u)
mat = np.zeros((N, M, dofs_Q0))
mat_int = np.zeros((N,M))
basis_func_u = fem.Function(V)
basis_func_P = fem.Function(S)
for i in range(N):
    basis_func_u.x.array[:] = basis_u[:, i]
    for j in range(M):
        basis_func_P.x.array[:] = basis_P[:, j]
        # this contains the integrated contributions
        mat_int[i, j] = fem.assemble_scalar(
            fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx))
        # this contains the cell-wise integrated contributions
        mat[i, j, :] = fem.petsc.assemble_vector(
            fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * cell_avg * dx)).array.copy()
# assert that the integrated contributions match the sum of the cell-wise contributions
assert(np.allclose(mat.sum(axis=2), mat_int))

# reshape mat to (N*M, dofs_Q0) and mat_int to (N*M,) for the linear program
A = mat.reshape(-1, dofs_Q0)
A = np.concatenate((A, np.expand_dims(weights, 0)))  # volume constraint
b = mat_int.reshape(-1)
b = np.concatenate((b, np.array([np.sum(weights)])))  # volume constraint
assert(np.allclose(A @ np.ones(dofs_Q0), b))

# ECM
from scipy.optimize import nnls
from scipy.linalg import lstsq
from dolfinx import mesh as dmesh
from dolfinx.io import VTXWriter, XDMFFile

# so this contains the factors of each integrated cell contribution
magic_points, magic_weights = my_ecm(A, b, tol=1e-6)
new_weights = magic_weights * weights[magic_points]  # re-normalize by the cell volume to get the correct weighting

# create meshtags with value 999 to mark active cells 
tdim = mesh.topology.dim
mesh.topology.create_entities(tdim)
indices = np.asarray(magic_points, dtype=np.int32)
indices = np.unique(indices)
values  = 999*np.ones_like(indices, dtype=np.int32)
cell_tags = dmesh.meshtags(mesh, tdim, indices, values)

# visualize the active cells
with XDMFFile(MPI.COMM_WORLD, "active.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_tags, mesh.geometry)

# create a measure that integrates only over the active cells with dx_hr(999)
dx_hr = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)


omega = fem.Function(Q0)
omega.x.array[:] = 0.0
omega.x.array[magic_points] = new_weights / weights[magic_points]  # re-normalize by the cell volume to get the correct weighting
vol_ecm = fem.assemble_scalar(fem.form(omega * dx_hr(999)))
assert(np.isclose(volume, vol_ecm))

for i in range(N):
    basis_func_u.x.array[:] = basis_u[:, i]
    for j in range(M):
        basis_func_P.x.array[:] = basis_P[:, j]
        int_full = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx))
        int_ecm = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * omega * dx_hr(999)))
        assert(np.isclose(int_full, int_ecm))

# time full and ecm integrations
from time import time
from tqdm import tqdm
time_full = time()
for i in tqdm(range(10000)):
    i = np.random.randint(0, N)
    j = np.random.randint(0, M)
    basis_func_u.x.array[:] = basis_u[:, i]
    basis_func_P.x.array[:] = basis_P[:, j]
    int_full = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx))
print(f"Full integration: {time() - time_full}s")

time_ecm = time()
for k in tqdm(range(10000)):
    i = np.random.randint(0, N)
    j = np.random.randint(0, M)
    basis_func_u.x.array[:] = basis_u[:, i]
    basis_func_P.x.array[:] = basis_P[:, j]
    int_ecm = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * omega * dx_hr(999)))
print(f"ECM integration: {time() - time_ecm}s")

# Variant: create a submesh of the active cells and transfer the basis functions to it, then integrate over the submesh with the standard measure dx_sub
def parent_to_sub_array(arr_parent, V_parent, V_sub, cell_map):
    """
    Copy dof values from a parent-mesh array into a submesh array, by cell.

    Parameters
    ----------
    arr_parent : (n_parent_dofs,) or (n_parent_dofs, k) ndarray
        Source data laid out like Function.x.array on the parent space.
    V_parent, V_sub : dolfinx.fem.FunctionSpace
        Spaces with matching element on parent mesh and submesh.
    cell_map : array_like of int
        cell_map[c_sub] = local parent-cell index for submesh cell c_sub
        (returned by dolfinx.mesh.create_submesh).

    Returns
    -------
    arr_sub : the same array that was passed in, now filled.
    """
    bs_p = V_parent.dofmap.bs
    bs_s = V_sub.dofmap.bs
    assert bs_p == bs_s, "block sizes of the two spaces must match"
    bs = bs_p

    cell_map = np.asarray(cell_map, dtype=np.int32)
    n_sub_cells = cell_map.size

    # Per-cell block-dof tables.
    pd = np.stack([V_parent.dofmap.cell_dofs(int(c)) for c in cell_map])
    sd = np.stack([V_sub.dofmap.cell_dofs(c)         for c in range(n_sub_cells)])
    assert pd.shape == sd.shape, "element layout differs between the two spaces"

    # Expand block (node) indices -> scalar dof indices.
    def expand(d):
        return (d[..., None] * bs + np.arange(bs)).reshape(d.shape[0], -1)

    pd_u = expand(pd).ravel()
    sd_u = expand(sd).ravel()

    # Works for 1D (single field) and 2D (a stack of k fields, columns).
    arr_sub = fem.Function(V_sub).x.array.copy()
    arr_sub[sd_u, ...] = arr_parent[pd_u, ...]
    return arr_sub

submesh, cell_map, vertex_map, geom_map = dmesh.create_submesh(
    mesh, tdim, indices
)
dx_sub = ufl.Measure("dx", domain=submesh)
V_sub = fem.functionspace(submesh, ("Lagrange", degree, (gdim,)))
S_sub = fem.functionspace(submesh, ("DG", 1, (gdim, gdim)))
Q0_sub = fem.functionspace(submesh, ("DG", 0))
basis_func_u_sub = fem.Function(V_sub)
basis_func_P_sub = fem.Function(S_sub)
omega_sub = fem.Function(Q0_sub)

basis_u_sub = []
for i in range(N):
    basis_u_sub.append(parent_to_sub_array(basis_u[:, i], V, V_sub, cell_map))

basis_P_sub = []
for j in range(M):
    basis_P_sub.append(parent_to_sub_array(basis_P[:, j], S, S_sub, cell_map))

omega_sub.x.array[:] = parent_to_sub_array(omega.x.array, Q0, Q0_sub, cell_map)

for i in range(N):
    basis_func_u.x.array[:] = basis_u[:, i]
    basis_func_u_sub.x.array[:] = basis_u_sub[i]
    for j in range(M):
        basis_func_P.x.array[:] = basis_P[:, j]
        basis_func_P_sub.x.array[:] = basis_P_sub[j]
        int_full = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx))
        int_ecm = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P_sub, ufl.grad(basis_func_u_sub)) * omega_sub * dx_sub))
        assert(np.isclose(int_full, int_ecm))

time_ecm = time()
for k in tqdm(range(10000)):
    i = np.random.randint(0, N)
    j = np.random.randint(0, M)
    basis_func_u_sub.x.array[:] = basis_u_sub[i]
    basis_func_P_sub.x.array[:] = basis_P_sub[j]
    int_ecm = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P_sub, ufl.grad(basis_func_u_sub)) * omega_sub * dx_sub))
print(f"ECM submesh integration: {time() - time_ecm}s")

# Variant: Write new mesh file and transfer
import gmsh
import numpy as np

tdim = submesh.topology.dim
assert submesh.topology.cell_type.name == "triangle"
# Sanity check that we really have P2 geometry:
n_nodes_per_cell = submesh.geometry.dofmap.shape[1]
assert n_nodes_per_cell == 6, f"expected triangle6, got {n_nodes_per_cell} nodes/cell"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)        # silence stdout
gmsh.model.add("active")

# 1) Create the discrete entity FIRST and use its tag everywhere below
ent = gmsh.model.addDiscreteEntity(tdim)            # returns a positive tag

# 2) Nodes
pts = submesh.geometry.x                            # (n_pts, 3); already 3D
node_tags = np.arange(1, pts.shape[0] + 1, dtype=np.int64)
gmsh.model.mesh.addNodes(
    dim=tdim,
    tag=ent,                                        # <-- the fix
    nodeTags=node_tags,
    coord=pts.reshape(-1).astype(np.float64),
)

# 3) Elements: triangle6 = type 9, with node permutation
GMSH_TRIANGLE6 = 9
perm = np.array([0, 1, 2, 5, 3, 4], dtype=np.int32) # DOLFINx -> Gmsh

conn = np.asarray(submesh.geometry.dofmap, dtype=np.int64)[:, perm]
conn = (conn + 1).reshape(-1)                       # Gmsh is 1-indexed

n_cells_local = submesh.topology.index_map(tdim).size_local
elem_tags = np.arange(1, n_cells_local + 1, dtype=np.int64)

gmsh.model.mesh.addElementsByType(ent, GMSH_TRIANGLE6, elem_tags, conn)
pg = gmsh.model.addPhysicalGroup(tdim, [ent], tag=1)  # explicit tag is safer
gmsh.model.setPhysicalName(tdim, pg, "active")

gmsh.write("active.msh")
gmsh.finalize()

mesh_2, _, _ = io.gmshio.read_from_msh("active.msh", comm, 0, gdim=gdim)
dx_2 = ufl.Measure("dx", domain=mesh_2)

V_2 = fem.functionspace(mesh_2, ("Lagrange", degree, (gdim,)))
S_2 = fem.functionspace(mesh_2, ("DG", 1, (gdim, gdim)))
Q0_2 = fem.functionspace(mesh_2, ("DG", 0))
basis_func_u_2 = fem.Function(V_2)
basis_func_P_2 = fem.Function(S_2)
omega_2 = fem.Function(Q0_2)

import numpy as np
from scipy.spatial import cKDTree
from dolfinx import fem

# ── 1. cell chain: mesh2 → submesh → parent ────────────────────────────────

def cell_centroids(msh):
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    return msh.geometry.x[msh.geometry.dofmap[:n]].mean(axis=1)  # (n_cells, 3)

_, mesh2_to_sub = cKDTree(cell_centroids(submesh)).query(
    cell_centroids(mesh_2), k=1
)
mesh2_to_parent = cell_map[mesh2_to_sub.astype(np.int32)]   # (n_cells_mesh2,)

# ── 2. Continuous / DG-0: coordinate permutation (correct for unique coords) ──

def make_coord_perm(V_src, V_dst, tol=1e-10):
    xs, xd = V_src.tabulate_dof_coordinates(), V_dst.tabulate_dof_coordinates()
    dist, perm = cKDTree(xs).query(xd, k=1)
    assert dist.max() < tol * (1 + np.abs(xs).max()), \
        f"dof match failed, max dist = {dist.max():.2e}"
    return perm.astype(np.int64)

def apply_coord_perm(arr, perm, bs):
    src = (perm[:, None] * bs + np.arange(bs)).ravel()
    return arr[src].copy()

perm_V  = make_coord_perm(V,  V_2)    # Lagrange: one global dof per node  ✓
perm_Q0 = make_coord_perm(Q0, Q0_2)   # DG-0: centroid is unique per cell  ✓

# ── 3. DG-1 (S): cell-based permutation — fixes the ambiguity ──────────────

def make_dg_perm(V_src, V_dst, src_cells_for_dst):
    """
    Build (src_scalar_indices, dst_scalar_indices) for a DG function transfer.

    Strategy
    --------
    1. Match cells by centroid  →  src_cells_for_dst[c_dst] = c_src
    2. Within each matched cell pair, match local block-dofs by their
       physical coordinate (brute-force, n_local is tiny: 3 for P1-tri).
    3. Expand block → scalar indices via dofmap.bs.
    """
    bs = V_src.dofmap.bs
    assert bs == V_dst.dofmap.bs, "block sizes must match"

    n_dst   = src_cells_for_dst.size
    n_local = V_dst.dofmap.cell_dofs(0).size       # block-dofs per cell

    xs = V_src.tabulate_dof_coordinates()           # (n_block_dofs_src, 3)
    xd = V_dst.tabulate_dof_coordinates()           # (n_block_dofs_dst, 3)

    all_src = np.array([V_src.dofmap.cell_dofs(int(c))
                        for c in src_cells_for_dst])  # (n_dst, n_local)
    all_dst = np.array([V_dst.dofmap.cell_dofs(c)
                        for c in range(n_dst)])        # (n_dst, n_local)

    if n_local > 1:
        # Vectorised pairwise distance: (n_dst, n_local_src, n_local_dst, 3)
        diff  = xs[all_src][:, :, None, :] - xd[all_dst][:, None, :, :]
        dists = np.linalg.norm(diff, axis=3)           # (n_dst, nl, nl)

        # For each dst local dof, find the nearest src local dof
        lp = np.argmin(dists, axis=1)                  # (n_dst, n_local)
        all_src = all_src[np.arange(n_dst)[:, None], lp]

    src_flat = (all_src[:, :, None] * bs + np.arange(bs)).reshape(-1).astype(np.int64)
    dst_flat = (all_dst[:, :, None] * bs + np.arange(bs)).reshape(-1).astype(np.int64)
    return src_flat, dst_flat

src_S, dst_S = make_dg_perm(S, S_2, mesh2_to_parent)
_S2_size     = fem.Function(S_2).x.array.size          # pre-allocated once

def apply_dg_perm(arr, src_flat, dst_flat, out_size):
    result = np.zeros(out_size, dtype=arr.dtype)
    result[dst_flat] = arr[src_flat]
    return result

# ── 4. Transfer all modes (offline, done once) ─────────────────────────────

basis_u_2 = [apply_coord_perm(basis_u[:, i], perm_V, V.dofmap.bs)
             for i in range(N)]

basis_P_2 = [apply_dg_perm(basis_P[:, j], src_S, dst_S, _S2_size)
             for j in range(M)]

omega_2.x.array[:] = apply_coord_perm(omega.x.array, perm_Q0, Q0.dofmap.bs)
omega_2.x.scatter_forward()

# ── 5. Quick sanity check on cell 0 before the big loop ───────────────────

c2 = 0
cp = int(mesh2_to_parent[c2])
bs = S.dofmap.bs

def cell_vals(V, arr, c):
    d = V.dofmap.cell_dofs(c)
    return arr[(d[:, None] * bs + np.arange(bs)).ravel()]

print("DG-1 transfer ok:",
      np.allclose(cell_vals(S,   basis_P[:, 0], cp),
                  cell_vals(S_2, basis_P_2[0],  c2)))

# ── 6. Integration loop ────────────────────────────────────────────────────

for i in range(N):
    basis_func_u.x.array[:]   = basis_u[:, i]
    basis_func_u_2.x.array[:] = basis_u_2[i]
    for j in range(M):
        basis_func_P.x.array[:]   = basis_P[:, j]
        basis_func_P_2.x.array[:] = basis_P_2[j]

        int_full = fem.assemble_scalar(fem.form(
            ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx
        ))
        int_ecm = fem.assemble_scalar(fem.form(
            ufl.inner(basis_func_P_2, ufl.grad(basis_func_u_2)) * omega_2 * dx_2
        ))
        assert np.allclose(int_full, int_ecm, rtol=1e-8)


time_ecm = time()
for k in tqdm(range(10000)):
    i = np.random.randint(0, N)
    j = np.random.randint(0, M)
    basis_func_u_2.x.array[:] = basis_u_2[i]
    basis_func_P_2.x.array[:] = basis_P_2[j]
    int_ecm = fem.assemble_scalar(fem.form(ufl.inner(basis_func_P_2, ufl.grad(basis_func_u_2)) * omega_2 * dx_2))
print(f"ECM mesh2 integration: {time() - time_ecm}s")