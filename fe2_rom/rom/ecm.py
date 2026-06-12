from dolfinx import io, fem, mesh as dmesh
from dolfinx.fem import petsc
from dolfinx.io import XDMFFile
from mpi4py import MPI
from time import time
import numpy as np
import scipy.sparse as sp
import ufl
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from tqdm import tqdm
import gmsh


# ── Sparse helper ─────────────────────────────────────────────────────────────

def petsc_to_scipy(mat):
    """Convert a PETSc matrix to a scipy.sparse.csr_matrix."""
    ai, aj, av = mat.getValuesCSR()
    m, n = mat.getSize()
    return sp.csr_matrix((av, aj, ai), shape=(m, n))


# ── Built-in ECM algorithm (replaceable by user) ──────────────────────────────

def my_ecm(A, b, tol=1e-4, candidate_batch=None, seed=0, max_points=None,
           backward_prune=True):
    """Empirical cubature method via greedy non-negative pursuit.

    Greedy max-correlation selection with a Lawson-Hanson-style
    non-negativity repair. The per-iteration least-squares solve uses a
    Cholesky factorisation of the (k x k) Gram matrix of the selected
    columns — O(n_rows·k + k³) per iteration instead of the O(n_rows·k²)
    of a fresh SVD-based ``lstsq``, which dominates on tall constraint
    matrices once a few hundred points are selected.

    Parameters
    ----------
    A : (n_constraints, n_candidates) ndarray
    b : (n_constraints,) ndarray
    tol : relative residual tolerance
    candidate_batch : int or None
        If set, each greedy step scores only this many randomly-chosen
        remaining candidates (stochastic greedy). The deterministic full
        scan (None, default) picks better points and is affordable unless
        n_constraints*n_candidates is very large.
    seed : int
        RNG seed for the candidate subsampling (used only when candidate_batch set).
    max_points : int or None
        Optional cap on the number of selected points (stop early).
    backward_prune : bool
        After convergence, repeatedly try to remove selected points that
        later additions have made redundant: drop the smallest-weight point,
        re-solve, and keep the removal while the residual stays below tol
        (greedy never revisits a point once added, so the final set usually
        contains some). Costs one Gram re-factorisation per attempted
        removal. Default True.

    Returns
    -------
    magic_points : list of int   — selected candidate indices
    alpha        : ndarray       — non-negative weights
    """
    print(f"A shape: {A.shape}, b shape: {b.shape}, tol: {tol}, candidate_batch: {candidate_batch}")
    n_rows, n_cand = A.shape
    rng = np.random.default_rng(seed)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0.0] = 1.0
    A_norm = A / col_norms
    b_norm = np.linalg.norm(b)

    cap = 256                                  # growing buffers for the selected set
    AS = np.empty((n_rows, cap))               # selected columns of A
    G = np.empty((cap, cap))                   # Gram matrix AS^T AS
    beta = np.empty(cap)                       # AS^T b
    S = []                                     # selected candidate indices
    in_S = np.zeros(n_cand, dtype=bool)
    alpha = np.empty(0)
    r = b.copy()
    k_iter, n_repair = 0, 0

    def grow():
        nonlocal cap, AS, G, beta
        AS = np.concatenate([AS, np.empty((n_rows, cap))], axis=1)
        G_new = np.empty((2 * cap, 2 * cap))
        G_new[:cap, :cap] = G
        G = G_new
        beta = np.concatenate([beta, np.empty(cap)])
        cap *= 2

    def solve(k):
        """Solve G[:k,:k] x = beta[:k]; jitter the diagonal if near-singular."""
        Gk = G[:k, :k]
        jitter = 0.0
        while True:
            try:
                return cho_solve(cho_factor(Gk + jitter * np.eye(k), lower=True), beta[:k])
            except np.linalg.LinAlgError:
                jitter = 1e-14 * np.trace(Gk) / k if jitter == 0.0 else 100.0 * jitter
                if jitter > np.trace(Gk):
                    raise

    def drop(keep):
        """Remove the selected entries with keep[pos] == False, compacting buffers."""
        kept = np.flatnonzero(keep)
        for pos in np.flatnonzero(~keep)[::-1]:
            idx = S.pop(int(pos))
            in_S[idx] = False
        m = kept.size
        AS[:, :m] = AS[:, kept]
        G[:m, :m] = G[np.ix_(kept, kept)]
        beta[:m] = beta[kept]
        return m

    print((k_iter, len(alpha), n_repair), 1.0)
    while np.linalg.norm(r) / b_norm > tol:
        if max_points is not None and len(S) >= max_points:
            print(f"ECM: max_points={max_points} reached, stopping early")
            break
        # 1. greedy selection by correlation with the residual
        if candidate_batch is not None and n_cand - len(S) > candidate_batch:
            cols = rng.choice(np.flatnonzero(~in_S), size=candidate_batch, replace=False)
            new = int(cols[np.argmax(A_norm[:, cols].T @ r)])
        else:
            scores = A_norm.T @ r
            scores[in_S] = -np.inf
            new = int(np.argmax(scores))
            if scores[new] <= 0.0:
                print("ECM: no candidate correlates with the residual, stopping early")
                break
        # 2. append column to the Gram system
        k = len(S)
        if k + 1 > cap:
            grow()
        a = A[:, new]
        AS[:, k] = a
        g = AS[:, :k].T @ a
        G[:k, k] = g
        G[k, :k] = g
        G[k, k] = a @ a
        beta[k] = a @ b
        S.append(new)
        in_S[new] = True
        k += 1
        # 3. solve, repairing negative weights (Lawson-Hanson inner loop)
        alpha_feas = np.concatenate([alpha, [0.0]])
        alpha_new = solve(k)
        while np.any(alpha_new < 0.0):
            n_repair += 1
            neg = alpha_new < 0.0
            t = alpha_feas[neg] / (alpha_feas[neg] - alpha_new[neg])
            theta = min(float(t.min()), 1.0)
            alpha_feas = alpha_feas + theta * (alpha_new - alpha_feas)
            keep = alpha_feas > 1e-12 * alpha_feas.max()
            if keep.all():
                keep[int(np.argmin(alpha_feas))] = False
            k = drop(keep)
            alpha_feas = np.clip(alpha_feas[keep], 0.0, None)
            alpha_new = solve(k)
        alpha = alpha_new
        r = b - AS[:, :k] @ alpha
        k_iter += 1
        print((k_iter, len(alpha), n_repair), np.linalg.norm(r) / b_norm)

    # ── backward pruning: drop points made redundant by later additions ─────
    # The greedy never revisits a point once added. Removing column j from an
    # unconstrained least-squares fit increases the squared residual by
    # alpha_j² / [G⁻¹]_jj; points whose predicted residual stays below tol are
    # removal candidates. Each removal is verified on trial copies (including
    # the non-negativity repair) before being committed.
    if backward_prune and len(S) > 1:
        n_removed = 0
        while len(S) > 1:
            k = len(S)
            try:
                cf = cho_factor(G[:k, :k], lower=True)
            except np.linalg.LinAlgError:
                break
            Ginv_diag = np.abs(np.diagonal(cho_solve(cf, np.eye(k))))
            sq_inc = alpha ** 2 / Ginv_diag
            pos = int(np.argmin(sq_inc))
            if np.linalg.norm(r) ** 2 + sq_inc[pos] > (tol * b_norm) ** 2:
                break
            # trial removal on copies, with Lawson-Hanson repair
            idx = np.delete(np.arange(k), pos)
            G_t = G[np.ix_(idx, idx)]
            beta_t = beta[idx]
            alpha_feas = np.clip(alpha[idx], 0.0, None)
            alpha_t = None
            while True:
                try:
                    alpha_t = cho_solve(cho_factor(G_t, lower=True), beta_t)
                except np.linalg.LinAlgError:
                    alpha_t = None
                    break
                if np.all(alpha_t >= 0.0):
                    break
                neg = alpha_t < 0.0
                t = alpha_feas[neg] / (alpha_feas[neg] - alpha_t[neg])
                theta = min(float(t.min()), 1.0)
                alpha_feas = alpha_feas + theta * (alpha_t - alpha_feas)
                keep_t = alpha_feas > 1e-12 * alpha_feas.max()
                if keep_t.all():
                    keep_t[int(np.argmin(alpha_feas))] = False
                idx = idx[keep_t]
                G_t = G_t[np.ix_(keep_t, keep_t)]
                beta_t = beta_t[keep_t]
                alpha_feas = np.clip(alpha_feas[keep_t], 0.0, None)
            if alpha_t is None:
                break
            r_t = b - AS[:, idx] @ alpha_t
            if np.linalg.norm(r_t) / b_norm > tol:
                break
            keep = np.zeros(k, dtype=bool)
            keep[idx] = True
            drop(keep)
            n_removed += k - len(S)
            alpha = alpha_t
            r = r_t
        if n_removed:
            print(f"backward prune: removed {n_removed} -> {len(S)} points, "
                  f"residual {np.linalg.norm(r) / b_norm:.3e}")
    return S, alpha


# ── Transfer utilities ────────────────────────────────────────────────────────

def _cell_map_to_array(cell_map, submesh):
    """Return a sub→parent cell index array from an EntityMap or pass-through ndarray."""
    if isinstance(cell_map, dmesh.EntityMap):
        tdim = submesh.topology.dim
        n_sub = submesh.topology.index_map(tdim).size_local
        return np.asarray(
            cell_map.sub_topology_to_topology(np.arange(n_sub, dtype=np.int32), inverse=False),
            dtype=np.int32,
        )
    return np.asarray(cell_map, dtype=np.int32)


def _parent_to_sub_array(arr_parent, V_parent, V_sub, cell_map):
    """Transfer a dof array from a parent-mesh space to the matching submesh space.

    Matches dofs cell-by-cell through `cell_map` (sub→parent cell indices);
    both spaces must have the same element and block size.
    """
    bs = V_parent.dofmap.bs
    assert bs == V_sub.dofmap.bs
    cell_map = _cell_map_to_array(cell_map, V_sub.mesh)
    n_sub = cell_map.size
    pd = np.stack([V_parent.dofmap.cell_dofs(int(c)) for c in cell_map])
    sd = np.stack([V_sub.dofmap.cell_dofs(c) for c in range(n_sub)])

    def expand(d):
        return (d[..., None] * bs + np.arange(bs)).reshape(d.shape[0], -1)

    arr_sub = np.zeros(V_sub.dofmap.index_map.size_local * V_sub.dofmap.index_map_bs)
    arr_sub[expand(sd).ravel()] = arr_parent[expand(pd).ravel()]
    return arr_sub


def _cell_centroids(msh):
    """Return the (n_local_cells, 3) array of cell geometry-node centroids."""
    tdim = msh.topology.dim
    n = msh.topology.index_map(tdim).size_local
    return msh.geometry.x[msh.geometry.dofmap[:n]].mean(axis=1)


def _make_coord_perm(V_src, V_dst, tol=1e-10):
    """Return perm such that dst dof k sits at the coordinates of src dof perm[k].

    Only valid for spaces with coordinate-unique dofs (e.g. Lagrange, DG-0);
    asserts that every dst dof has a src dof within `tol`.
    """
    xs, xd = V_src.tabulate_dof_coordinates(), V_dst.tabulate_dof_coordinates()
    dist, perm = cKDTree(xs).query(xd, k=1)
    assert dist.max() < tol * (1 + np.abs(xs).max()), f"dof match failed, max dist={dist.max():.2e}"
    return perm.astype(np.int64)


def _apply_coord_perm(arr, perm, bs):
    """Apply a dof permutation from :func:`_make_coord_perm` to a blocked dof array."""
    return arr[(perm[:, None] * bs + np.arange(bs)).ravel()].copy()


def _make_dg_perm(V_src, V_dst, src_cells_for_dst):
    """Build flat (src, dst) index arrays mapping DG dofs between meshes.

    For each dst cell c, `src_cells_for_dst[c]` names the matching src cell;
    within a cell, local dofs are paired by nearest coordinates (DG dofs are
    not coordinate-unique across cells, so :func:`_make_coord_perm` cannot be
    used). Apply with :func:`_apply_dg_perm`.
    """
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
    """Apply the (src, dst) index map from :func:`_make_dg_perm` to a dof array."""
    result = np.zeros(out_size, dtype=arr.dtype)
    result[dst_flat] = arr[src_flat]
    return result


def _write_submesh_to_gmsh(submesh, filename, comm, gdim):
    """Export a triangle6 submesh to a .msh file and reload it as a fresh mesh.

    Used by :meth:`ECM.test_variant3` to obtain a standalone gmsh-built mesh of
    the active cells. Only second-order triangles are supported.
    """
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
    return io.gmsh.read_from_msh(filename, comm, 0, gdim=gdim).mesh


# ── ECM ───────────────────────────────────────────────────────────────────────

class ECM:
    """Adapted Empirical Cubature Method for hyper-reduction.

    Finds a sparse, non-negative cell-wise quadrature rule (magic points +
    weights) that exactly reproduces, over the full mesh, a set of constraint
    integrals built from ROM bases: the virtual-work block ∫P:∇u dX, the
    average-stress block ∫P dX (component-wise), optional extra
    volume-integral blocks, and the total volume. Each block can be switched
    off independently (`include_uP`, `include_P_int`), e.g. to find an
    integration rule for ∫P dX alone or only for the `kwargs` blocks.

    Parameters
    ----------
    basis_u   : (n_dofs_V, N) ndarray   — displacement ROM basis.
                May be None when include_uP=False.
    basis_P   : (n_dofs_S, M) ndarray   — stress ROM basis.
                May be None when include_uP=False and include_P_int=False
                (e.g. an integration rule built from `kwargs` blocks only).
    V         : displacement FunctionSpace (None allowed iff basis_u is None)
    S         : stress FunctionSpace (None allowed iff basis_P is None)
    degree    : int — Lagrange degree used to rebuild V on the active-cell
                submesh in save_variant2 / the test variants (default 1).
    sigma_u   : (N,) array-like or None — singular values for u basis;
                rows are weighted by sigma_u[i]*sigma_P[j] / (sigma_u[0]*sigma_P[0]).
                If None, all u modes are weighted equally.
    sigma_P   : (M,) array-like or None — singular values for P basis (same logic).
    ratio_uP  : float — overall scale of the P·∇u constraint block relative to
                the volume row (default 1.0).
    ratio_P   : float — overall scale of the ∫P dX constraint block relative to
                the volume row (default 1.0).
    include_uP : bool — if False, drop the P·∇u constraint block entirely
                (keep only the ∫P dX block, any extra blocks, and the volume
                row), e.g. to find an integration rule for ∫P dX alone.
                Default True.
    include_P_int : bool — if False, drop the ∫P dX constraint block entirely
                (keep only the P·∇u block, any extra blocks, and the volume row).
                Default True.
    row_weight_tol : float or None — if set, drop (and skip assembling) every
                constraint row whose normalised singular-value weight falls
                below this value: su[i]*sp[j] for the P·∇u block, sp[j] for the
                ∫P block, sx[j] for extras (the block `ratio` is not included).
                Rows with weight below the ECM tolerance are numerically
                invisible to the greedy selection, but a more effective and
                self-consistent choice is the POD truncation level itself
                (e.g. ~1e-3 when the smallest kept σ/σ₀ is ~1e-3): mode pairs
                whose combined weight is below what POD already discarded.
                Saves assembly time, constraint-matrix memory, and ECM
                iteration cost, all linear in the rows dropped. None (default)
                keeps all rows. In the compressed path (`compress_uP`) the uP
                rows are instead pruned by relative row norm with the same
                threshold.
    compress_uP : int, float, "auto" or None — if set, build the P·∇u block by
                structured data compression (Liljegren-Sailer) instead of
                assembling all N*M rows: the σ-weighted quadrature-point data
                of the P modes is SVD-truncated to K_t "training stress
                modes" in the metric induced by the ∇u modes, and the block
                is materialised as N*K_t rows by tensor contraction — no
                per-pair PETSc assembly. int = K_t directly — the
                recommended form, using the paper's heuristic K_t ≈ expected
                number of magic points + a small margin. float = relative
                compression tolerance, K_t chosen as the smallest rank with
                κ ≤ compress_uP·‖R·Ĝ‖_F; note this tail is bounded below by
                roughly min(σ_P)/σ_P[0], so tolerances below that select
                full rank (no compression) — 1e-2 is realistic, 1e-4 usually
                is not. "auto" = fully automatic (requires compute_magic):
                K_t starts at the smallest rank with relative spectrum tail
                κ/‖R·Ĝ‖_F ≤ tol, the trained rule is verified against the
                exact full-system residual (evaluated only at the selected
                cells, so it stays cheap), and K_t is doubled and retrained
                if the verification fails. The verified residual is stored
                in ``self.true_residual``. The
                compression error κ is stored in ``self.kappa_uP``
                (cell-aggregated bound in ``self.kappa_uP_cells``); the true
                training residual obeys η ≤ η_t + κ_cells·‖ω − ω_full‖.
                None (default) keeps the exact per-pair assembly.
    quad_degree : int or None — quadrature degree used by the compressed
                path to tabulate the bases (default (deg_u − 1) + deg_P,
                exact on affine cells; raise it for curved geometry, e.g.
                +2 on triangle6 meshes).
    cell_chunk : int — cells per tabulation chunk in the compressed path;
                bounds peak memory (default 4096).
    kwargs    : dict or None — extra volume-integral-only constraint blocks,
                ``name -> {"basis": (n_dofs_X, M_x) ndarray, "space":
                FunctionSpace, "sigma": optional (M_x,) singular values,
                "ratio": optional float block scale}``. For each block, every
                flat component of every mode contributes a row ∫X_j_c dX.

    Usage
    -----
    ``ecm = ECM(...)``; ``ecm.compute_magic(tol=...)``; then
    ``ecm.save_variant2(out_dir)`` for the online stage, or
    ``ecm.test_variant{1,2,3}()`` to verify/benchmark the rule.

    Call patterns
    -------------
    Any combination of the three block types is allowed; the volume row is
    always included. Bases/spaces of disabled blocks may be passed as None
    (basis_u/V are only needed for the uP block; basis_P/S for the uP and
    ∫P blocks).

    P·∇u + ∫P (default)::

        ECM(basis_u, basis_P, V, S)

    P·∇u + ∫P + extras::

        ECM(basis_u, basis_P, V, S,
            kwargs={"W": {"basis": basis_W, "space": SW}})

    P·∇u only::

        ECM(basis_u, basis_P, V, S, include_P_int=False)

    P·∇u + extras::

        ECM(basis_u, basis_P, V, S, include_P_int=False,
            kwargs={"W": {"basis": basis_W, "space": SW}})

    ∫P only (integration rule for the average stress alone)::

        ECM(None, basis_P, None, S, include_uP=False)

    ∫P + extras::

        ECM(None, basis_P, None, S, include_uP=False,
            kwargs={"W": {"basis": basis_W, "space": SW}})

    extras only (rule built purely from the kwargs blocks)::

        ECM(None, None, None, None, include_uP=False, include_P_int=False,
            kwargs={"W": {"basis": basis_W, "space": SW}})

    Large problems — structured compression of the P·∇u block
    (no per-pair assembly, N*K_t instead of N*M rows)::

        ECM(basis_u, basis_P, V, S, sigma_u=su, sigma_P=sp,
            compress_uP=1e-4)        # or an explicit rank: compress_uP=30
    """

    def __init__(self, basis_u, basis_P, V, S,
                 degree: int = 1,
                 sigma_u=None, sigma_P=None,
                 ratio_uP=1.0, ratio_P=1.0,
                 include_uP: bool = True,
                 include_P_int: bool = True,
                 row_weight_tol: float | None = None,
                 compress_uP: int | float | None = None,
                 quad_degree: int | None = None,
                 cell_chunk: int = 4096,
                 kwargs: dict | None = None):
        if include_uP:
            assert basis_u is not None and V is not None, \
                "basis_u and V are required when include_uP=True"
        if include_uP or include_P_int:
            assert basis_P is not None and S is not None, \
                "basis_P and S are required when include_uP or include_P_int is True"
        self.basis_u = basis_u
        self.basis_P = basis_P
        self.N = 0 if basis_u is None else basis_u.shape[1]
        self.M = 0 if basis_P is None else basis_P.shape[1]
        self.V = V
        self.S = S
        spaces = [sp_ for sp_ in (V, S) if sp_ is not None]
        spaces += [spec["space"] for spec in (kwargs or {}).values()]
        assert spaces, "need at least one function space to determine the mesh"
        self.mesh = spaces[0].mesh
        self.gdim = self.mesh.topology.dim
        self.degree = degree
        self._Q0 = fem.functionspace(self.mesh, ("DG", 0))
        self._weights, self._volume = self._assemble_weights()
        self.sigma_u = np.ones(self.N) if sigma_u is None else np.asarray(sigma_u, dtype=float)
        self.sigma_P = np.ones(self.M) if sigma_P is None else np.asarray(sigma_P, dtype=float)
        self.ratio_uP = ratio_uP
        self.ratio_P = ratio_P
        self._include_uP = include_uP
        self._include_P_int = include_P_int
        self.row_weight_tol = row_weight_tol
        self.compress_uP = compress_uP
        self.quad_degree = quad_degree
        self.cell_chunk = cell_chunk
        self.kappa_uP = None
        self.kappa_uP_cells = None
        self.true_residual = None
        self._auto_tol = None        # set by compute_magic when compress_uP == "auto"
        self._auto_K_t_floor = 1     # raised on auto-retry
        self._K_t_used = None
        self._n_uP_rows = 0
        self._uP_tab = None          # quadrature tabulation cache (compressed path)
        self._b_uP_full = None       # exact full-system uP rhs (compressed path)
        # Extra volume-integral-only bases.  Each entry is
        #   name -> {"basis", "space", "sigma" (optional), "ratio" (optional)}
        self.extras: dict = {}
        for name, spec in (kwargs or {}).items():
            basis = spec["basis"]
            sigma = spec.get("sigma")
            self.extras[name] = {
                "basis": basis,
                "space": spec["space"],
                "M": basis.shape[1],
                "sigma": (np.ones(basis.shape[1]) if sigma is None
                          else np.asarray(sigma, dtype=float)),
                "ratio": float(spec.get("ratio", 1.0)),
            }
        self.magic_points = None
        self.magic_weights = None

    def _assemble_weights(self):
        """Return (per-cell volumes as a DG-0 vector, total mesh volume)."""
        dx = ufl.dx(domain=self.mesh)
        cell_avg = ufl.TestFunction(self._Q0)
        w = petsc.assemble_vector(fem.form(cell_avg * dx)).array.copy()
        return w, float(w.sum())

    @property
    def weights(self):
        """Per-cell volumes (the exact DG-0 quadrature weights of the full mesh)."""
        return self._weights

    @property
    def volume(self):
        """Total mesh volume (sum of `weights`)."""
        return self._volume

    def _build_matrices(self):
        """Assemble the constraint matrix A and rhs b.

        Rows: N*M (if include_uP) + M*n_P_components (if include_P_int)
        + the extra blocks from `kwargs` + 1 volume row. If `row_weight_tol`
        is set, rows whose normalised singular-value weight falls below it are
        pruned before assembly (their forms are never assembled).
        """
        dofs_Q0 = self._Q0.dofmap.index_map.size_global * self._Q0.dofmap.index_map_bs
        dx = ufl.dx(domain=self.mesh)
        cell_avg = ufl.TestFunction(self._Q0)

        A_blocks, b_blocks = [], []
        theta = 0.0 if self.row_weight_tol is None else self.row_weight_tol

        self._n_uP_rows = 0
        if self._include_uP and self.compress_uP is not None:
            A_uP = self._compressed_uP_block()
            A_blocks.append(A_uP)
            b_blocks.append(A_uP.sum(axis=1))
            self._n_uP_rows = A_uP.shape[0]
        elif self._include_uP:
            # Per-row weights from normalised singular values
            # sigma_u[0] and sigma_P[0] are the reference (largest) values
            su = self.sigma_u / self.sigma_u[0]   # shape (N,), su[0] == 1
            sp = self.sigma_P / self.sigma_P[0]   # shape (M,), sp[0] == 1
            w_rel = np.outer(su, sp)              # (N, M), w_rel[0, 0] == 1
            keep = w_rel >= theta
            if self.row_weight_tol is not None:
                print(f"uP block: keeping {keep.sum()}/{keep.size} rows "
                      f"(row_weight_tol={theta:g})")

            basis_func_u = fem.Function(self.V)
            basis_func_P = fem.Function(self.S)
            form = fem.form(ufl.inner(basis_func_P, ufl.grad(basis_func_u)) * cell_avg * dx)
            rows, w_rows = [], []
            for i in range(self.N):
                if not keep[i].any():
                    continue
                basis_func_u.x.array[:] = self.basis_u[:, i]
                for j in range(self.M):
                    if not keep[i, j]:
                        continue
                    basis_func_P.x.array[:] = self.basis_P[:, j]
                    rows.append(petsc.assemble_vector(form).array.copy())
                    # row weight: ratio_uP * su[i] * sp[j]
                    w_rows.append(self.ratio_uP * w_rel[i, j])
            if rows:
                mat = np.asarray(rows) * np.asarray(w_rows)[:, np.newaxis]
                A_blocks.append(mat)
                b_blocks.append(mat.sum(axis=1))
                self._n_uP_rows = mat.shape[0]

        if self._include_P_int:
            sp = self.sigma_P / self.sigma_P[0]   # shape (M,), sp[0] == 1
            keep_P = sp >= theta
            if self.row_weight_tol is not None:
                print(f"P-int block: keeping {keep_P.sum()}/{keep_P.size} modes "
                      f"(row_weight_tol={theta:g})")

            basis_func_P = fem.Function(self.S)
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

            rows, w_rows = [], []
            for j in np.flatnonzero(keep_P):
                basis_func_P.x.array[:] = self.basis_P[:, j]
                for form_Pc in p_component_forms:
                    rows.append(petsc.assemble_vector(form_Pc).array.copy())
                    # row weight: ratio_P * sp[j] (same for all components)
                    w_rows.append(self.ratio_P * sp[j])
            if rows:
                mat_P = np.asarray(rows) * np.asarray(w_rows)[:, np.newaxis]
                A_blocks.append(mat_P)
                b_blocks.append(mat_P.sum(axis=1))

        # Extra volume-integral-only blocks from `kwargs`.
        for name, entry in self.extras.items():
            basis = entry["basis"]
            space = entry["space"]
            M_x = entry["M"]
            sigma = entry["sigma"]
            ratio = entry["ratio"]
            sx = sigma / sigma[0]                          # (M_x,), sx[0] == 1
            keep_x = sx >= theta
            if self.row_weight_tol is not None:
                print(f"{name} block: keeping {keep_x.sum()}/{keep_x.size} modes "
                      f"(row_weight_tol={theta:g})")

            basis_func_x = fem.Function(space)
            x_shape = basis_func_x.ufl_shape
            x_component_forms = []
            for idx in (np.ndindex(*x_shape) if x_shape else [()]):
                if not idx:
                    comp = basis_func_x
                elif len(idx) == 1:
                    comp = basis_func_x[idx[0]]
                else:
                    comp = basis_func_x[idx]
                x_component_forms.append(fem.form(comp * cell_avg * dx))

            rows, w_rows = [], []
            for j in np.flatnonzero(keep_x):
                basis_func_x.x.array[:] = basis[:, j]
                for form_xc in x_component_forms:
                    rows.append(petsc.assemble_vector(form_xc).array.copy())
                    # row weight: ratio * sx[j] (same for all components)
                    w_rows.append(ratio * sx[j])
            if rows:
                mat_x = np.asarray(rows) * np.asarray(w_rows)[:, np.newaxis]
                A_blocks.append(mat_x)
                b_blocks.append(mat_x.sum(axis=1))

        A = np.vstack(A_blocks + [self._weights])
        b = np.concatenate(b_blocks + [[self._volume]])
        assert np.allclose(A @ np.ones(dofs_Q0), b)
        return A, b

    def _compressed_uP_block(self):
        """Build the P·∇u block via structured data compression.

        Implements the preprocessing of Liljegren-Sailer (structured
        compression of empirical-quadrature training data): instead of
        assembling all N*M rows A[(i,j), m] = w[i,j] ∫_{Ω_m} P_j : ∇u_i dX,
        the σ_P-weighted quadrature-point data Ĝ of the P modes is compressed
        by a truncated SVD in the metric R² (diagonal, from the analytic QR
        of the structured factor: R²_{(q,c)} = ratio_uP² W_q² Σ_i (su_i
        [∇u_i]_c(ξ_q))²). The truncated right singular vectors define K_t
        "training stress modes" Ĝ·V₁, and the block is materialised as N*K_t
        rows by tensor contraction over quadrature points — the bases are
        tabulated with `fem.Expression`, no per-pair PETSc assembly.

        Stores the Frobenius compression error ``self.kappa_uP`` =
        √(Σ_{i>K_t} σ_i²) and the cell-aggregated bound
        ``self.kappa_uP_cells`` = √(n_q·n_comp)·κ (the constraint matrices
        before/after compression differ by at most this in Frobenius norm).
        """
        import basix
        import basix.ufl

        cell_name = self.mesh.topology.cell_type.name
        deg_u = self.V.ufl_element().degree
        deg_P = self.S.ufl_element().degree
        qdeg = self.quad_degree if self.quad_degree is not None else (deg_u - 1) + deg_P
        pts, wts = basix.make_quadrature(getattr(basix.CellType, cell_name), qdeg)
        n_q = wts.size
        n_cells = self._Q0.dofmap.index_map.size_global * self._Q0.dofmap.index_map_bs

        # Scaled quadrature weights W[cell, q] = w_q |detJ(ξ_q)|, tabulated by
        # assembling per-point indicator functions in a matching quadrature space
        q_el = basix.ufl.quadrature_element(cell_name, value_shape=(),
                                            scheme="default", degree=qdeg)
        Q = fem.functionspace(self.mesh, q_el)
        ind = fem.Function(Q)
        cell_avg = ufl.TestFunction(self._Q0)
        dxq = ufl.dx(domain=self.mesh,
                     metadata={"quadrature_scheme": "default", "quadrature_degree": qdeg})
        form_w = fem.form(ind * cell_avg * dxq)
        W = np.empty((n_cells, n_q))
        for q in range(n_q):
            ind.x.array[:] = 0.0
            ind.x.array[q::n_q] = 1.0
            W[:, q] = petsc.assemble_vector(form_w).array
        assert np.isclose(W.sum(), self._volume), "quadrature weight tabulation failed"

        su = self.sigma_u / self.sigma_u[0]
        sp = self.sigma_P / self.sigma_P[0]

        basis_func_u = fem.Function(self.V)
        basis_func_P = fem.Function(self.S)
        expr_gu = fem.Expression(ufl.grad(basis_func_u), pts)
        expr_P = fem.Expression(basis_func_P, pts)
        n_comp = int(np.prod(basis_func_P.ufl_shape))
        assert n_comp == int(np.prod(ufl.grad(basis_func_u).ufl_shape))

        chunks = [np.arange(s, min(s + self.cell_chunk, n_cells), dtype=np.int32)
                  for s in range(0, n_cells, self.cell_chunk)]

        def tab(expr, func, basis, n_modes, cells_):
            out = np.empty((n_modes, cells_.size, n_q, n_comp))
            for k in range(n_modes):
                func.x.array[:] = basis[:, k]
                out[k] = expr.eval(self.mesh, cells_).reshape(cells_.size, n_q, n_comp)
            return out

        # Pass 1: Gram matrix H = Ĝᵀ R² Ĝ of the σ_P-weighted P data
        # (BLAS-friendly: H += B Bᵀ with B = Ĝ·R per cell chunk), plus the
        # exact full-system rhs b_uP[(i,j)] = w[i,j] ∫_Ω P_j : ∇u_i dX
        H = np.zeros((self.M, self.M))
        B_int = np.zeros((self.N, self.M))
        for cells_ in tqdm(chunks, desc="uP compression: Gram"):
            Uc = tab(expr_gu, basis_func_u, self.basis_u, self.N, cells_)
            r2 = (self.ratio_uP ** 2) * (W[cells_] ** 2)[:, :, None] \
                 * np.tensordot(su ** 2, Uc ** 2, axes=(0, 0))
            Gc = tab(expr_P, basis_func_P, self.basis_P, self.M, cells_) \
                 * sp[:, None, None, None]
            B = (Gc * np.sqrt(r2)).reshape(self.M, -1)
            H += B @ B.T
            Uw = (Uc * W[cells_][None, :, :, None]).reshape(self.N, -1)
            B_int += Uw @ Gc.reshape(self.M, -1).T
        self._b_uP_full = (self.ratio_uP * su[:, None] * B_int).ravel()
        self._uP_tab = {"pts": pts, "W": W, "n_q": n_q, "n_comp": n_comp,
                        "su": su, "sp": sp}

        evals, evecs = np.linalg.eigh(H)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        evecs = evecs[:, order]
        tails = np.sqrt(np.cumsum(evals[::-1])[::-1])   # tails[k] = ‖σ_{≥k}‖

        if isinstance(self.compress_uP, str):
            assert self.compress_uP == "auto", f"unknown compress_uP={self.compress_uP!r}"
            assert self._auto_tol is not None, \
                "compress_uP='auto' picks K_t from the ECM tolerance — use compute_magic()"
            # deliberately small prior, capped even when the spectrum tail is
            # flat: the verify-and-grow loop in compute_magic corrects
            # underestimates (an overestimate would never be corrected down)
            target = self._auto_tol * tails[0]
            K_t = int(np.searchsorted(-tails, -target))
            K_t = min(K_t, max(16, self.M // 8))
            K_t = max(min(max(K_t, self._auto_K_t_floor), self.M), 1)
        elif isinstance(self.compress_uP, (int, np.integer)):
            K_t = int(min(self.compress_uP, self.M))
        else:
            rel_tol = float(self.compress_uP)
            K_t = int(np.searchsorted(-tails, -rel_tol * tails[0]))
            K_t = max(min(K_t, self.M), 1)
        self._K_t_used = K_t
        self.kappa_uP = float(np.sqrt(evals[K_t:].sum()))
        self.kappa_uP_cells = self.kappa_uP * np.sqrt(n_q * n_comp)
        print(f"uP compression: K_t={K_t}/{self.M} training stress modes, "
              f"rows {self.N * self.M} -> {self.N * K_t}, "
              f"kappa={self.kappa_uP:.3e} (cell-aggregated {self.kappa_uP_cells:.3e})")

        # Pass 2: materialise A_t[(i,t), m] = ratio_uP su_i Σ_{q∈m,c} W U_i Ĝ·V₁
        # (batched GEMM per cell: (n_cells, N, n_q·n_comp) @ (n_cells, n_q·n_comp, K_t))
        Vt = evecs[:, :K_t]
        A_t = np.empty((self.N, K_t, n_cells))
        for cells_ in tqdm(chunks, desc="uP compression: rows"):
            nc = cells_.size
            Uc = tab(expr_gu, basis_func_u, self.basis_u, self.N, cells_)
            Gc = tab(expr_P, basis_func_P, self.basis_P, self.M, cells_) \
                 * sp[:, None, None, None]
            Gt = np.tensordot(Vt, Gc, axes=(0, 0))           # (K_t, nc, q, c)
            Gt *= W[cells_][None, :, :, None]
            U_b = Uc.reshape(self.N, nc, -1).transpose(1, 0, 2)
            G_b = Gt.reshape(K_t, nc, -1).transpose(1, 2, 0)
            A_t[:, :, cells_] = np.matmul(U_b, G_b).transpose(1, 2, 0)
        A_t *= self.ratio_uP * su[:, None, None]
        A_t = A_t.reshape(self.N * K_t, n_cells)

        if self.row_weight_tol is not None:
            row_norms = np.linalg.norm(A_t, axis=1)
            keep = row_norms >= self.row_weight_tol * row_norms.max()
            print(f"uP compressed block: keeping {keep.sum()}/{keep.size} rows "
                  f"(row_weight_tol={self.row_weight_tol:g} on relative row norms)")
            A_t = A_t[keep]
        return A_t

    def _full_residual(self, A, b, magic_points, weights):
        """Exact relative residual of a sparse rule on the FULL constraint system.

        The uP block is evaluated against all N*M uncompressed rows — but only
        their columns at the selected cells are needed, so they are rebuilt by
        quadrature tabulation at those cells (cheap, no PETSc assembly and no
        (N·M) x n_cells matrix). The remaining rows of `A` (∫P block, extras,
        volume row) are uncompressed already and evaluated directly. Requires a
        preceding compressed build (uses the tabulation data it cached).
        """
        assert self._uP_tab is not None and self._b_uP_full is not None, \
            "_full_residual requires a compressed uP build"
        pts = self._uP_tab["pts"]
        W = self._uP_tab["W"]
        n_q, n_comp = self._uP_tab["n_q"], self._uP_tab["n_comp"]
        su, sp = self._uP_tab["su"], self._uP_tab["sp"]

        basis_func_u = fem.Function(self.V)
        basis_func_P = fem.Function(self.S)
        expr_gu = fem.Expression(ufl.grad(basis_func_u), pts)
        expr_P = fem.Expression(basis_func_P, pts)

        points = np.asarray(magic_points, dtype=np.int32)
        weights = np.asarray(weights)
        r_uP = self._b_uP_full.copy()
        for s in range(0, points.size, self.cell_chunk):
            cells_ = points[s:s + self.cell_chunk]
            w_ = weights[s:s + self.cell_chunk]
            U_S = np.empty((self.N, cells_.size, n_q, n_comp))
            for i in range(self.N):
                basis_func_u.x.array[:] = self.basis_u[:, i]
                U_S[i] = expr_gu.eval(self.mesh, cells_).reshape(cells_.size, n_q, n_comp)
            G_S = np.empty((self.M, cells_.size, n_q, n_comp))
            for j in range(self.M):
                basis_func_P.x.array[:] = self.basis_P[:, j]
                G_S[j] = expr_P.eval(self.mesh, cells_).reshape(cells_.size, n_q, n_comp)
            G_S *= sp[:, None, None, None]
            Uw = (U_S * W[cells_][None, :, :, None]).reshape(self.N, cells_.size, -1)
            # A_cols[(i,j), s] = ratio_uP su_i Σ_{q,c} W U_i G_j
            cols = np.matmul(Uw.transpose(1, 0, 2),
                             G_S.reshape(self.M, cells_.size, -1).transpose(1, 2, 0))
            cols = cols.transpose(1, 2, 0) * (self.ratio_uP * su[:, None, None])
            r_uP -= cols.reshape(self.N * self.M, cells_.size) @ w_

        omega = np.zeros(A.shape[1])
        omega[points] = weights
        r_other = b[self._n_uP_rows:] - A[self._n_uP_rows:] @ omega
        num = np.sqrt(np.linalg.norm(r_uP) ** 2 + np.linalg.norm(r_other) ** 2)
        den = np.sqrt(np.linalg.norm(self._b_uP_full) ** 2
                      + np.linalg.norm(b[self._n_uP_rows:]) ** 2)
        return float(num / den)

    def compute_magic(self, ecm_func=None, tol=1e-6, max_auto_rounds=4, **ecm_func_kwargs):
        """Compute magic points and weights.

        Parameters
        ----------
        ecm_func : callable(A, b, tol, **kw) -> (magic_points, alpha), default my_ecm
        tol      : relative residual tolerance passed to ecm_func
        max_auto_rounds : int — with ``compress_uP="auto"``, maximum number of
            train-verify-retry rounds (K_t doubles each retry).
        **ecm_func_kwargs : forwarded to ecm_func — e.g. ``candidate_batch=100``
            (and optional ``seed``) to use the stochastic-greedy subset selection
            in :func:`my_ecm`.

        Results are stored in ``self.magic_points`` (active cell indices) and
        ``self.magic_weights`` (their non-negative quadrature weights). With
        ``compress_uP="auto"``, K_t is selected from `tol`, the trained rule is
        verified against the exact full-system residual, K_t is doubled and the
        training repeated if verification fails; the verified residual is
        stored in ``self.true_residual``.
        """
        if ecm_func is None:
            ecm_func = my_ecm
        auto = self._include_uP and isinstance(self.compress_uP, str)
        if auto:
            self._auto_tol = tol
            self._auto_K_t_floor = 1
        for round_ in range(max_auto_rounds if auto else 1):
            A, b = self._build_matrices()
            self.magic_points, self.magic_weights = ecm_func(A, b, tol=tol, **ecm_func_kwargs)
            if not auto:
                return
            self.true_residual = self._full_residual(A, b, self.magic_points, self.magic_weights)
            print(f"auto-K_t round {round_ + 1}: K_t={self._K_t_used}, "
                  f"{len(self.magic_points)} points, "
                  f"true full-system residual {self.true_residual:.3e} (tol {tol:g})")
            if self.true_residual <= tol or self._K_t_used >= self.M:
                return
            # paper's rule (K_t = M_c + 10) with the measured point count,
            # guarded by geometric growth; saturates at M (= exact system)
            self._auto_K_t_floor = int(min(self.M, max(len(self.magic_points) + 10,
                                                       1.5 * self._K_t_used)))
        print("auto-K_t: max_auto_rounds reached, keeping last result")

    def show_active_cells(self, filename="active.xdmf"):
        """Write the selected (active) cells as meshtags to an XDMF file for ParaView."""
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
        """Return (meshtags marking the active cells with tag 999, sorted unique cell indices)."""
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_entities(tdim)
        indices = np.unique(np.asarray(self.magic_points, dtype=np.int32))
        values = 999 * np.ones_like(indices, dtype=np.int32)
        return dmesh.meshtags(self.mesh, tdim, indices, values), indices

    def _make_omega(self):
        """Return the DG-0 weight function: magic weights on active cells, 0 elsewhere."""
        omega = fem.Function(self._Q0)
        omega.x.array[:] = 0.0
        omega.x.array[self.magic_points] = self.magic_weights
        return omega

    def save_variant2(self, output_dir):
        """Build the variant-2 submesh data and save to output_dir.

        Saves
        -----
        indices.npy         : (n_active,) int32  — parent cell indices of active cells
        omega_sub.npy       : (n_dofs_Q0_sub,)   — ECM weight function on submesh
        basis_u_sub.npy     : (n_dofs_V_sub, N)  — displacement modes on submesh
        basis_u.npy         : (n_dofs_V, N)      — full-mesh displacement modes
        (the two basis_u files are skipped when basis_u is None)

        The submesh can be reconstructed at load time via
            submesh, cell_map, _, _ = dmesh.create_submesh(mesh, tdim, indices)
        """
        import os
        assert self.magic_points is not None, "Call compute_magic first"
        os.makedirs(output_dir, exist_ok=True)

        _, indices = self._make_cell_tags_and_indices()
        tdim = self.mesh.topology.dim
        submesh, cell_map, _, _ = dmesh.create_submesh(self.mesh, tdim, indices)

        Q0_sub = fem.functionspace(submesh, ("DG", 0))

        omega = self._make_omega()
        omega_sub = _parent_to_sub_array(omega.x.array, self._Q0, Q0_sub, cell_map)

        np.save(os.path.join(output_dir, "indices.npy"),   indices)
        np.save(os.path.join(output_dir, "omega_sub.npy"), omega_sub)

        if self.basis_u is not None:
            V_sub = fem.functionspace(submesh, ("Lagrange", self.degree, (self.gdim,)))
            basis_u_sub = np.stack([
                _parent_to_sub_array(self.basis_u[:, i], self.V, V_sub, cell_map)
                for i in range(self.N)
            ]).T
            np.save(os.path.join(output_dir, "basis_u_sub.npy"), basis_u_sub)
            np.save(os.path.join(output_dir, "basis_u.npy"), self.basis_u)

    # ── integration test variants ─────────────────────────────────────────────

    def test_variant1(self, n_trials=10000):
        """Verify and time the ECM rule on the full mesh.

        Integrates omega * dx_hr(999) (the weight function on a measure
        restricted to the active cells), asserts every ∫P:∇u mode pair matches
        the full integral, then benchmarks full vs ECM assembly over
        `n_trials` random mode pairs. Requires basis_u and basis_P.
        """
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
        """Verify and time the ECM rule on an active-cell submesh.

        Builds the submesh with create_submesh, transfers the bases and weight
        function to it, asserts every ∫P:∇u mode pair matches the full-mesh
        integral, then benchmarks full vs submesh assembly. This is the layout
        saved by :meth:`save_variant2`. Requires basis_u and basis_P.
        """
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
        """Verify and time the ECM rule on a gmsh round-tripped active-cell mesh.

        Exports the active-cell submesh to .msh, reloads it as a standalone
        mesh, transfers the bases by coordinate/cell matching, asserts every
        ∫P:∇u mode pair matches the full-mesh integral, then benchmarks full
        vs remeshed assembly. Triangle6 meshes only (see
        :func:`_write_submesh_to_gmsh`). Requires basis_u and basis_P.
        """
        assert self.magic_points is not None, "Call compute_magic first"
        dx = ufl.dx(domain=self.mesh)
        _, indices = self._make_cell_tags_and_indices()
        tdim = self.mesh.topology.dim
        submesh, cell_map, _, _ = dmesh.create_submesh(self.mesh, tdim, indices)
        cell_map_arr = _cell_map_to_array(cell_map, submesh)

        omega = self._make_omega()
        mesh_2 = _write_submesh_to_gmsh(submesh, "active.msh", self.mesh.comm, self.gdim)
        dx_2 = ufl.Measure("dx", domain=mesh_2)

        V_2  = fem.functionspace(mesh_2, ("Lagrange", self.degree, (self.gdim,)))
        S_2  = fem.functionspace(mesh_2, ("DG", 1, (self.gdim, self.gdim)))
        Q0_2 = fem.functionspace(mesh_2, ("DG", 0))

        mesh2_to_sub    = cKDTree(_cell_centroids(submesh)).query(_cell_centroids(mesh_2), k=1)[1]
        mesh2_to_parent = cell_map_arr[mesh2_to_sub.astype(np.int32)]

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
    from fe2_rom.rom.pod import POD

    comm = MPI.COMM_WORLD
    gdim = 2
    degree = 2

    mesh = io.gmsh.read_from_msh("holes.msh", comm, 0, gdim=gdim).mesh
    V  = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    S  = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))
    S0 = fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))

    energy_tol = 0.9999

    snapshots_u = POD.load_snapshots("output/snapshots/u_fluc_*.npy")
    pod_u = POD(snapshots_u, V, inner_product="H1")
    N = pod_u.n_modes(energy_tol)

    snapshots_P = POD.load_snapshots("output/snapshots/P_*.npy")
    pod_P = POD(snapshots_P, S, inner_product="L2")
    M = pod_P.n_modes(energy_tol)

    print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes")

    ecm = ECM(pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
              degree=degree,
              sigma_u=np.sqrt(pod_u.eigenvalues[:N]), sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
              ratio_uP=2.0, ratio_P=1.0)
    ecm.compute_magic(tol=1e-6)

    print("Number of magic points:", len(ecm.magic_points))
