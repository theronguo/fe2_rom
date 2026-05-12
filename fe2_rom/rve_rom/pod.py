from dolfinx import io, fem, mesh as dmesh
from dolfinx.fem import petsc
from dolfinx.io import XDMFFile
from mpi4py import MPI
from glob import glob
from time import time
import numpy as np
import scipy.sparse as sp
import ufl
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.linalg import lstsq
from scipy.spatial import cKDTree
from tqdm import tqdm
import gmsh

# ── Sparse helper ─────────────────────────────────────────────────────────────

def petsc_to_scipy(mat):
    ai, aj, av = mat.getValuesCSR()
    m, n = mat.getSize()
    return sp.csr_matrix((av, aj, ai), shape=(m, n))


# ── Built-in ECM algorithm (replaceable by user) ──────────────────────────────

def my_ecm(A, b, tol=1e-4):
    """Empirical cubature method via greedy NNLS pursuit.

    Parameters
    ----------
    A : (n_constraints, n_candidates) ndarray
    b : (n_constraints,) ndarray
    tol : relative residual tolerance

    Returns
    -------
    magic_points : list of int   — selected candidate indices
    alpha        : ndarray       — non-negative weights
    """
    candidates = list(range(A.shape[1]))
    magic_points = []
    alpha = np.array([])
    r = b.copy()
    A_norm = A / np.linalg.norm(A, axis=0)
    k, counter_nnls = 0, 0
    print((k, len(alpha), counter_nnls), np.linalg.norm(r) / np.linalg.norm(b))
    while np.linalg.norm(r) / np.linalg.norm(b) > tol:
        new_idx = int(np.argmax(A_norm[:, candidates].T @ r))
        magic_points.append(candidates.pop(new_idx))
        A_current = A[:, magic_points]
        alpha, _, _, _ = lstsq(A_current, b)
        if np.any(alpha < 0):
            alpha = nnls(A_current, b, maxiter=1000000)[0]
            counter_nnls += 1
            for idx in np.where(alpha < 1e-8)[0][::-1]:
                candidates.append(magic_points.pop(idx))
            A_current = A[:, magic_points]
            alpha, _, _, _ = lstsq(A_current, b)
        r = b - A_current @ alpha
        k += 1
        print((k, len(alpha), counter_nnls), np.linalg.norm(r) / np.linalg.norm(b))
    return magic_points, alpha


# ── Transfer utilities ────────────────────────────────────────────────────────

def _parent_to_sub_array(arr_parent, V_parent, V_sub, cell_map):
    bs = V_parent.dofmap.bs
    assert bs == V_sub.dofmap.bs
    cell_map = np.asarray(cell_map, dtype=np.int32)
    n_sub = cell_map.size
    pd = np.stack([V_parent.dofmap.cell_dofs(int(c)) for c in cell_map])
    sd = np.stack([V_sub.dofmap.cell_dofs(c) for c in range(n_sub)])

    def expand(d):
        return (d[..., None] * bs + np.arange(bs)).reshape(d.shape[0], -1)

    arr_sub = np.zeros(V_sub.dofmap.index_map.size_local * V_sub.dofmap.index_map_bs)
    arr_sub[expand(sd).ravel()] = arr_parent[expand(pd).ravel()]
    return arr_sub


def _cell_centroids(msh):
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    return msh.geometry.x[msh.geometry.dofmap[:n]].mean(axis=1)


def _make_coord_perm(V_src, V_dst, tol=1e-10):
    xs, xd = V_src.tabulate_dof_coordinates(), V_dst.tabulate_dof_coordinates()
    dist, perm = cKDTree(xs).query(xd, k=1)
    assert dist.max() < tol * (1 + np.abs(xs).max()), f"dof match failed, max dist={dist.max():.2e}"
    return perm.astype(np.int64)


def _apply_coord_perm(arr, perm, bs):
    return arr[(perm[:, None] * bs + np.arange(bs)).ravel()].copy()


def _make_dg_perm(V_src, V_dst, src_cells_for_dst):
    bs = V_src.dofmap.bs
    assert bs == V_dst.dofmap.bs
    n_dst = src_cells_for_dst.size
    n_local = V_dst.dofmap.cell_dofs(0).size
    xs = V_src.tabulate_dof_coordinates()
    xd = V_dst.tabulate_dof_coordinates()
    all_src = np.array([V_src.dofmap.cell_dofs(int(c)) for c in src_cells_for_dst])
    all_dst = np.array([V_dst.dofmap.cell_dofs(c) for c in range(n_dst)])
    if n_local > 1:
        diff = xs[all_src][:, :, None, :] - xd[all_dst][:, None, :, :]
        lp = np.argmin(np.linalg.norm(diff, axis=3), axis=1)
        all_src = all_src[np.arange(n_dst)[:, None], lp]
    src_flat = (all_src[:, :, None] * bs + np.arange(bs)).reshape(-1).astype(np.int64)
    dst_flat = (all_dst[:, :, None] * bs + np.arange(bs)).reshape(-1).astype(np.int64)
    return src_flat, dst_flat


def _apply_dg_perm(arr, src_flat, dst_flat, out_size):
    result = np.zeros(out_size, dtype=arr.dtype)
    result[dst_flat] = arr[src_flat]
    return result


def _write_submesh_to_gmsh(submesh, filename, comm, gdim):
    tdim = submesh.topology.dim
    assert submesh.topology.cell_type.name == "triangle"
    assert submesh.geometry.dofmap.shape[1] == 6, \
        f"expected triangle6, got {submesh.geometry.dofmap.shape[1]} nodes/cell"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("active")
    ent = gmsh.model.addDiscreteEntity(tdim)
    pts = submesh.geometry.x
    node_tags = np.arange(1, pts.shape[0] + 1, dtype=np.int64)
    gmsh.model.mesh.addNodes(tdim, ent, node_tags, pts.reshape(-1).astype(np.float64))
    perm = np.array([0, 1, 2, 5, 3, 4], dtype=np.int32)  # DOLFINx -> Gmsh node order
    conn = (np.asarray(submesh.geometry.dofmap, dtype=np.int64)[:, perm] + 1).reshape(-1)
    n_cells = submesh.topology.index_map(tdim).size_local
    gmsh.model.mesh.addElementsByType(ent, 9, np.arange(1, n_cells + 1, dtype=np.int64), conn)
    pg = gmsh.model.addPhysicalGroup(tdim, [ent], tag=1)
    gmsh.model.setPhysicalName(tdim, pg, "active")
    gmsh.write(filename)
    gmsh.finalize()
    return io.gmshio.read_from_msh(filename, comm, 0, gdim=gdim)[0]


# ── POD ───────────────────────────────────────────────────────────────────────

class POD:
    """Proper Orthogonal Decomposition via correlation matrix and eigendecomposition.

    Parameters
    ----------
    snapshots     : (n_snapshots, n_dofs) ndarray
    V             : dolfinx FunctionSpace the snapshots live in
    inner_product : "L2" or "H1"
    """

    def __init__(self, snapshots, V, inner_product="L2"):
        self.snapshots = snapshots
        self.V = V
        self.mesh = V.mesh
        self._ip_matrix = self._assemble_inner_product_matrix(inner_product)
        self.basis, self.eigenvalues = self._compute()

    @staticmethod
    def load_snapshots(pattern):
        files = [f for f in sorted(glob(pattern)) if "dof_coords" not in f]
        return np.array([np.load(f) for f in files])

    @staticmethod
    def load_and_align_snapshots(pattern, V):
        """Load snapshots and permute DOFs to match V's serial DOF ordering.

        When snapshots are saved from a parallel run (mpirun -np N), the DOF
        ordering in each .npy file follows the parallel global-DOF-index
        assignment, which differs from the serial DOF ordering used by V.

        If a companion *_dof_coords.npy file was written alongside the
        snapshots (by the solver's _save_snapshot method), this function reads
        those coordinates, matches them to V's DOF coordinates via a KD-tree,
        and permutes the snapshot arrays accordingly.  If no coords file is
        found the snapshots are returned as-is (correct for serial saves).
        """
        import os, re
        files = [f for f in sorted(glob(pattern)) if "dof_coords" not in f]
        snapshots = np.array([np.load(f) for f in files])
        if not files:
            return snapshots

        # Derive coords filename: strip trailing _<timestamp>.npy
        first = files[0]
        coords_basename = re.sub(r'_[\d]+\.[\d]+\.npy$', '_dof_coords.npy',
                                  os.path.basename(first))
        coords_path = os.path.join(os.path.dirname(first), coords_basename)
        if not os.path.exists(coords_path):
            return snapshots

        saved_coords = np.load(coords_path)       # (n_nodes, gdim) — parallel ordering
        serial_coords = V.tabulate_dof_coordinates()  # (n_nodes, gdim) — serial ordering
        bs = V.dofmap.index_map_bs

        _, perm = cKDTree(saved_coords).query(serial_coords, k=1)
        dof_perm = (perm[:, None] * bs + np.arange(bs)).ravel()
        return snapshots[:, dof_perm]

    def _assemble_inner_product_matrix(self, inner_product):
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dx = ufl.dx(domain=self.mesh)
        if inner_product == "H1":
            expr = ufl.inner(u, v) * dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        else:
            expr = ufl.inner(u, v) * dx
        return fem.assemble_matrix(fem.form(expr)).to_scipy()

    def _compute(self):
        S = self.snapshots
        HS = self._ip_matrix @ S.T        # (n_dofs, n_snapshots)
        C = S @ HS                         # (n_snapshots, n_snapshots) correlation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        basis = S.T @ eigenvectors[:, idx]
        return basis, eigenvalues[idx]

    def plot_eigenvalues(self):
        plt.semilogy(self.eigenvalues, marker="o")
        plt.xlabel("Mode")
        plt.ylabel("Eigenvalue")
        plt.title("POD Eigenvalues")
        plt.show()

    def n_modes(self, energy_fraction=0.9999):
        """Return the number of modes needed to capture *energy_fraction* of total energy."""
        cumsum = np.cumsum(self.eigenvalues)
        return int(np.searchsorted(cumsum, energy_fraction * cumsum[-1])) + 1

    def visualize_modes(self, n_modes, filename, visualization_space, interpolation_space=None):
        fn = fem.Function(visualization_space)
        fn_out = fem.Function(interpolation_space) if interpolation_space is not None else fn
        with XDMFFile(self.mesh.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            for i in range(n_modes):
                fn.x.array[:] = self.basis[:, i]
                if interpolation_space is not None:
                    fn_out.interpolate(fn)
                xdmf.write_function(fn_out, i)


# ── ECM ───────────────────────────────────────────────────────────────────────

class ECM:
    """Adapted Empirical Cubature Method for hyper-reduction.

    Parameters
    ----------
    basis_u   : (n_dofs_V, N) ndarray   — displacement ROM basis
    basis_P   : (n_dofs_S, M) ndarray   — stress ROM basis
    V         : displacement FunctionSpace
    S         : stress FunctionSpace
    sigma_u   : (N,) array-like or None — singular values for u basis;
                rows are weighted by sigma_u[i]*sigma_P[j] / (sigma_u[0]*sigma_P[0]).
                If None, all u modes are weighted equally.
    sigma_P   : (M,) array-like or None — singular values for P basis (same logic).
    ratio_uP  : float — overall scale of the P·∇u constraint block relative to
                the volume row (default 1.0).
    ratio_P   : float — overall scale of the ∫P dX constraint block relative to
                the volume row (default 1.0).
    """

    def __init__(self, basis_u, basis_P, V, S,
                 degree: int = 1,
                 sigma_u=None, sigma_P=None,
                 ratio_uP=1.0, ratio_P=1.0):
        self.basis_u = basis_u
        self.basis_P = basis_P
        self.N = basis_u.shape[1]
        self.M = basis_P.shape[1]
        self.V = V
        self.S = S
        self.mesh = V.mesh
        self.gdim = V.mesh.topology.dim
        self.degree = degree
        self._Q0 = fem.functionspace(self.mesh, ("DG", 0))
        self._weights, self._volume = self._assemble_weights()
        self.sigma_u = np.ones(self.N) if sigma_u is None else np.asarray(sigma_u, dtype=float)
        self.sigma_P = np.ones(self.M) if sigma_P is None else np.asarray(sigma_P, dtype=float)
        self.ratio_uP = ratio_uP
        self.ratio_P = ratio_P
        self.magic_points = None
        self.magic_weights = None

    def _assemble_weights(self):
        dx = ufl.dx(domain=self.mesh)
        cell_avg = ufl.TestFunction(self._Q0)
        w = petsc.assemble_vector(fem.form(cell_avg * dx)).array.copy()
        return w, float(w.sum())

    @property
    def weights(self):
        return self._weights

    @property
    def volume(self):
        return self._volume

    def _build_matrices(self):
        """Assemble the (N*M + M*n_P_components + 1) x dofs_Q0 constraint matrix A and rhs b."""
        dofs_Q0 = self._Q0.dofmap.index_map.size_global * self._Q0.dofmap.index_map_bs
        dx = ufl.dx(domain=self.mesh)
        cell_avg = ufl.TestFunction(self._Q0)
        basis_func_u = fem.Function(self.V)
        basis_func_P = fem.Function(self.S)
        
        mat = np.zeros((self.N, self.M, dofs_Q0))
        form = fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * cell_avg * dx)
        for i in range(self.N):
            basis_func_u.x.array[:] = self.basis_u[:, i]
            for j in range(self.M):
                basis_func_P.x.array[:] = self.basis_P[:, j]
                mat[i, j, :] = petsc.assemble_vector(
                    form
                ).array.copy()
        mat_int = mat.sum(axis=2)

        # Pre-compile one form per flat component of P for ∫P_j_c φ_k dX
        p_shape = basis_func_P.ufl_shape
        p_component_forms = []
        for idx in np.ndindex(*p_shape):
            if not idx:
                comp = basis_func_P
            elif len(idx) == 1:
                comp = basis_func_P[idx[0]]
            else:
                comp = basis_func_P[idx]
            p_component_forms.append(fem.form(comp * cell_avg * dx))
        n_P_components = len(p_component_forms)

        mat_P = np.zeros((self.M, n_P_components, dofs_Q0))
        for j in range(self.M):
            basis_func_P.x.array[:] = self.basis_P[:, j]
            for c, form_Pc in enumerate(p_component_forms):
                mat_P[j, c, :] = petsc.assemble_vector(form_Pc).array.copy()
        mat_P_int = mat_P.sum(axis=2)

        # Per-row weights from normalised singular values
        # sigma_u[0] and sigma_P[0] are the reference (largest) values
        su = self.sigma_u / self.sigma_u[0]   # shape (N,), su[0] == 1
        sp = self.sigma_P / self.sigma_P[0]   # shape (M,), sp[0] == 1

        # mat block: w[i,j] = ratio_uP * su[i] * sp[j]
        w_uP = self.ratio_uP * np.outer(su, sp)          # (N, M)
        mat_w     = mat     * w_uP[:, :, np.newaxis]     # (N, M, dofs_Q0)
        mat_int_w = mat_int * w_uP                        # (N, M)

        # mat_P block: w[j] = ratio_P * sp[j]  (same for all components c)
        w_P     = self.ratio_P * sp                       # (M,)
        mat_P_w     = mat_P     * w_P[:, np.newaxis, np.newaxis]   # (M, n_P_comp, dofs_Q0)
        mat_P_int_w = mat_P_int * w_P[:, np.newaxis]               # (M, n_P_comp)

        A = np.vstack([mat_w.reshape(-1, dofs_Q0), mat_P_w.reshape(-1, dofs_Q0), self._weights])
        b = np.concatenate([mat_int_w.reshape(-1), mat_P_int_w.reshape(-1), [self._volume]])
        assert np.allclose(A @ np.ones(dofs_Q0), b)
        return A, b

    def compute_magic(self, ecm_func=None, tol=1e-6):
        """Compute magic points and weights.

        Parameters
        ----------
        ecm_func : callable(A, b, tol) -> (magic_points, alpha), default my_ecm
        tol      : relative residual tolerance passed to ecm_func
        """
        if ecm_func is None:
            ecm_func = my_ecm
        A, b = self._build_matrices()
        self.magic_points, self.magic_weights = ecm_func(A, b, tol=tol)
        # self.magic_points = np.array(list(range(len(self.weights)))) if self.magic_points is None else self.magic_points
        # self.magic_weights = 1 if self.magic_weights is None else self.magic_weights

    def show_active_cells(self, filename="active.xdmf"):
        assert self.magic_points is not None, "Call compute_magic first"
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim)
        indices = np.unique(np.asarray(self.magic_points, dtype=np.int32))
        values = 999 * np.ones_like(indices, dtype=np.int32)
        cell_tags = dmesh.meshtags(self.mesh, tdim, indices, values)
        with XDMFFile(self.mesh.comm, filename, "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_meshtags(cell_tags, self.mesh.geometry)
        return cell_tags

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_cell_tags_and_indices(self):
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim)
        indices = np.unique(np.asarray(self.magic_points, dtype=np.int32))
        values = 999 * np.ones_like(indices, dtype=np.int32)
        return dmesh.meshtags(self.mesh, tdim, indices, values), indices

    def _make_omega(self):
        omega = fem.Function(self._Q0)
        omega.x.array[:] = 0.0
        omega.x.array[self.magic_points] = self.magic_weights
        return omega

    def save_variant2(self, output_dir):
        """Build the variant-2 submesh data and save to output_dir.

        Saves
        -----
        indices.npy         : (n_active,) int32  — parent cell indices of active cells
        basis_u_sub.npy     : (N, n_dofs_V_sub)  — displacement modes on submesh
        basis_P_sub.npy     : (M, n_dofs_S_sub)  — stress modes on submesh
        omega_sub.npy       : (n_dofs_Q0_sub,)   — ECM weight function on submesh

        The submesh can be reconstructed at load time via
            submesh, cell_map, _, _ = dmesh.create_submesh(mesh, tdim, indices)
        """
        import os
        assert self.magic_points is not None, "Call compute_magic first"
        os.makedirs(output_dir, exist_ok=True)

        _, indices = self._make_cell_tags_and_indices()
        tdim = self.mesh.topology.dim
        submesh, cell_map, _, _ = dmesh.create_submesh(self.mesh, tdim, indices)

        V_sub  = fem.functionspace(submesh, ("Lagrange", self.degree, (self.gdim,)))
        S_sub  = fem.functionspace(submesh, ("DG", 1, (self.gdim, self.gdim)))
        Q0_sub = fem.functionspace(submesh, ("DG", 0))

        basis_u_sub = np.stack([
            _parent_to_sub_array(self.basis_u[:, i], self.V, V_sub, cell_map)
            for i in range(self.N)
        ]).T
        basis_P_sub = np.stack([
            _parent_to_sub_array(self.basis_P[:, j], self.S, S_sub, cell_map)
            for j in range(self.M)
        ]).T

        omega = self._make_omega()
        omega_sub = _parent_to_sub_array(omega.x.array, self._Q0, Q0_sub, cell_map)

        np.save(os.path.join(output_dir, "indices.npy"),     indices)
        np.save(os.path.join(output_dir, "basis_u_sub.npy"), basis_u_sub)
        np.save(os.path.join(output_dir, "omega_sub.npy"),   omega_sub)
        np.save(os.path.join(output_dir, "basis_u.npy"), self.basis_u)
        np.save(os.path.join(output_dir, "basis_P.npy"), self.basis_P)

    # ── integration test variants ─────────────────────────────────────────────

    def test_variant1(self, n_trials=10000):
        """Full mesh: weight function omega * dx_hr(999) restricted to active cells."""
        assert self.magic_points is not None, "Call compute_magic first"
        dx = ufl.dx(domain=self.mesh)
        cell_tags, _ = self._make_cell_tags_and_indices()
        dx_hr = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags)
        omega = self._make_omega()

        assert np.isclose(self._volume, fem.assemble_scalar(fem.form(omega * dx_hr(999))))

        basis_func_u = fem.Function(self.V)
        basis_func_P = fem.Function(self.S)
        integrand = ufl.inner(basis_func_P, ufl.grad(basis_func_u))
        form_full = fem.form(integrand * dx)
        form_ecm  = fem.form(integrand * omega * dx_hr(999))

        for i in range(self.N):
            basis_func_u.x.array[:] = self.basis_u[:, i]
            for j in range(self.M):
                basis_func_P.x.array[:] = self.basis_P[:, j]
                assert np.isclose(
                    fem.assemble_scalar(form_full),
                    fem.assemble_scalar(form_ecm),
                )

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 1 full"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u.x.array[:] = self.basis_u[:, i]
            basis_func_P.x.array[:] = self.basis_P[:, j]
            fem.assemble_scalar(form_full)
        print(f"Variant 1 full integration:  {time() - t0:.3f}s")

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 1 ECM"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u.x.array[:] = self.basis_u[:, i]
            basis_func_P.x.array[:] = self.basis_P[:, j]
            fem.assemble_scalar(form_ecm)
        print(f"Variant 1 ECM integration:   {time() - t0:.3f}s")

    def test_variant2(self, n_trials=10000):
        """Submesh: create_submesh of active cells, transfer bases, integrate over dx_sub."""
        assert self.magic_points is not None, "Call compute_magic first"
        dx = ufl.dx(domain=self.mesh)
        _, indices = self._make_cell_tags_and_indices()
        tdim = self.mesh.topology.dim
        submesh, cell_map, _, _ = dmesh.create_submesh(self.mesh, tdim, indices)
        dx_sub = ufl.Measure("dx", domain=submesh)

        V_sub  = fem.functionspace(submesh, ("Lagrange", self.degree, (self.gdim,)))
        S_sub  = fem.functionspace(submesh, ("DG", 1, (self.gdim, self.gdim)))
        Q0_sub = fem.functionspace(submesh, ("DG", 0))

        basis_u_sub = [_parent_to_sub_array(self.basis_u[:, i], self.V, V_sub, cell_map) for i in range(self.N)]
        basis_P_sub = [_parent_to_sub_array(self.basis_P[:, j], self.S, S_sub, cell_map) for j in range(self.M)]

        omega = self._make_omega()
        omega_sub = fem.Function(Q0_sub)
        omega_sub.x.array[:] = _parent_to_sub_array(omega.x.array, self._Q0, Q0_sub, cell_map)

        basis_func_u     = fem.Function(self.V)
        basis_func_P     = fem.Function(self.S)
        basis_func_u_sub = fem.Function(V_sub)
        basis_func_P_sub = fem.Function(S_sub)
        form_full    = fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * dx)
        form_ecm_sub = fem.form(ufl.inner(basis_func_P_sub, ufl.grad(basis_func_u_sub)) * omega_sub * dx_sub)

        for i in range(self.N):
            basis_func_u.x.array[:]     = self.basis_u[:, i]
            basis_func_u_sub.x.array[:] = basis_u_sub[i]
            for j in range(self.M):
                basis_func_P.x.array[:]     = self.basis_P[:, j]
                basis_func_P_sub.x.array[:] = basis_P_sub[j]
                assert np.isclose(
                    fem.assemble_scalar(form_full),
                    fem.assemble_scalar(form_ecm_sub),
                )        

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 2 full"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u.x.array[:] = self.basis_u[:, i]
            basis_func_P.x.array[:] = self.basis_P[:, j]
            fem.assemble_scalar(form_full)
        print(f"Variant 2 full integration:  {time() - t0:.3f}s")

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 2 ECM"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u_sub.x.array[:] = basis_u_sub[i]
            basis_func_P_sub.x.array[:] = basis_P_sub[j]
            fem.assemble_scalar(form_ecm_sub)
        print(f"Variant 2 ECM submesh:       {time() - t0:.3f}s")

    def test_variant3(self, n_trials=10000):
        """Gmsh remesh: export submesh via gmsh, reload as fresh mesh, transfer bases."""
        assert self.magic_points is not None, "Call compute_magic first"
        dx = ufl.dx(domain=self.mesh)
        _, indices = self._make_cell_tags_and_indices()
        tdim = self.mesh.topology.dim
        submesh, cell_map, _, _ = dmesh.create_submesh(self.mesh, tdim, indices)

        omega = self._make_omega()
        mesh_2 = _write_submesh_to_gmsh(submesh, "active.msh", self.mesh.comm, self.gdim)
        dx_2 = ufl.Measure("dx", domain=mesh_2)

        V_2  = fem.functionspace(mesh_2, ("Lagrange", self.degree, (self.gdim,)))
        S_2  = fem.functionspace(mesh_2, ("DG", 1, (self.gdim, self.gdim)))
        Q0_2 = fem.functionspace(mesh_2, ("DG", 0))

        mesh2_to_sub    = cKDTree(_cell_centroids(submesh)).query(_cell_centroids(mesh_2), k=1)[1]
        mesh2_to_parent = cell_map[mesh2_to_sub.astype(np.int32)]

        perm_V    = _make_coord_perm(self.V,   V_2)
        perm_Q0   = _make_coord_perm(self._Q0, Q0_2)
        src_S, dst_S = _make_dg_perm(self.S, S_2, mesh2_to_parent)
        S2_size   = fem.Function(S_2).x.array.size

        basis_u_2 = [_apply_coord_perm(self.basis_u[:, i], perm_V,  self.V.dofmap.bs)    for i in range(self.N)]
        basis_P_2 = [_apply_dg_perm(self.basis_P[:, j],   src_S, dst_S, S2_size)          for j in range(self.M)]
        omega_2   = fem.Function(Q0_2)
        omega_2.x.array[:] = _apply_coord_perm(omega.x.array, perm_Q0, self._Q0.dofmap.bs)
        omega_2.x.scatter_forward()

        bs = self.S.dofmap.bs
        def _cell_vals(V, arr, c):
            d = V.dofmap.cell_dofs(c)
            return arr[(d[:, None] * bs + np.arange(bs)).ravel()]

        assert np.allclose(
            _cell_vals(self.S, self.basis_P[:, 0], int(mesh2_to_parent[0])),
            _cell_vals(S_2,    basis_P_2[0],       0),
        ), "DG-1 transfer sanity check failed"

        basis_func_u   = fem.Function(self.V)
        basis_func_P   = fem.Function(self.S)
        basis_func_u_2 = fem.Function(V_2)
        basis_func_P_2 = fem.Function(S_2)
        form_full   = fem.form(ufl.inner(basis_func_P,   ufl.grad(basis_func_u))   * dx)
        form_ecm_2  = fem.form(ufl.inner(basis_func_P_2, ufl.grad(basis_func_u_2)) * omega_2 * dx_2)

        for i in range(self.N):
            basis_func_u.x.array[:]   = self.basis_u[:, i]
            basis_func_u_2.x.array[:] = basis_u_2[i]
            for j in range(self.M):
                basis_func_P.x.array[:]   = self.basis_P[:, j]
                basis_func_P_2.x.array[:] = basis_P_2[j]
                assert np.allclose(
                    fem.assemble_scalar(form_full),
                    fem.assemble_scalar(form_ecm_2),
                    rtol=1e-8,
                )

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 3 full"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u.x.array[:] = self.basis_u[:, i]
            basis_func_P.x.array[:] = self.basis_P[:, j]
            fem.assemble_scalar(form_full)
        print(f"Variant 3 full integration:  {time() - t0:.3f}s")

        t0 = time()
        for _ in tqdm(range(n_trials), desc="Variant 3 ECM"):
            i, j = np.random.randint(0, self.N), np.random.randint(0, self.M)
            basis_func_u_2.x.array[:] = basis_u_2[i]
            basis_func_P_2.x.array[:] = basis_P_2[j]
            fem.assemble_scalar(form_ecm_2)
        print(f"Variant 3 ECM gmsh mesh:     {time() - t0:.3f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    gdim = 2
    degree = 2

    mesh, _, _ = io.gmshio.read_from_msh("holes.msh", comm, 0, gdim=gdim)
    V  = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    S  = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))
    S0 = fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))

    energy_tol = 0.9999

    snapshots_u = POD.load_snapshots("output/snapshots/u_fluc_*.npy")
    pod_u = POD(snapshots_u, V, inner_product="H1")
    N = pod_u.n_modes(energy_tol)
    # pod_u.plot_eigenvalues()
    # pod_u.visualize_modes(N, "pod_mode_u.xdmf", V)

    snapshots_P = POD.load_snapshots("output/snapshots/P_*.npy")
    pod_P = POD(snapshots_P, S, inner_product="L2")
    M = pod_P.n_modes(energy_tol)
    # pod_P.plot_eigenvalues()
    # pod_P.visualize_modes(M, "pod_mode_P.xdmf", S, S0)

    print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes")

    ecm = ECM(pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
              degree=degree,
              sigma_u=np.sqrt(pod_u.eigenvalues[:N]), sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
              ratio_uP=2.0, ratio_P=1.0)
    ecm.compute_magic(tol=1e-6)
    # ecm.show_active_cells("active.xdmf")

    # ecm.test_variant1(10000)
    # ecm.test_variant2(10000)
    # ecm.test_variant3(10000)

    print("Number of magic points:", len(ecm.magic_points))

    ecm.save_variant2("ecm_variant2_data")