from dolfinx import fem
from dolfinx.io import XDMFFile
from glob import glob
import logging
import numpy as np
import ufl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

logger = logging.getLogger("fe2_rom.rom.pod")


class POD:
    """Proper Orthogonal Decomposition in an L²/H¹ inner product.

    Two solver paths produce identical modes (up to the truncation rank):

    * ``method="snapshots"`` (default) — the method of snapshots: form the
      ``(n_snapshots × n_snapshots)`` correlation matrix ``C = S M Sᵀ`` and take
      its full eigendecomposition. Cheap eigenproblem, but the two dense products
      with ``S`` cost ``O(n_dof · n_snapshots²)`` and the full basis is built.
    * ``method="randomized"`` — randomized M-weighted SVD. Computes only the top
      ``rank`` modes at ``O(n_dof · n_snapshots · (rank + oversampling))`` and
      never materializes the full spectrum or the dense weighted snapshot matrix.
      Much cheaper in both flops and memory when ``rank ≪ n_snapshots``.

    Parameters
    ----------
    snapshots        : (n_snapshots, n_dofs) ndarray
    V                : dolfinx FunctionSpace the snapshots live in
    inner_product    : "L2" or "H1"
    method           : "snapshots" (default) or "randomized"
    rank             : number of modes to compute (required for "randomized")
    oversampling     : extra sketch dimensions for the randomized range finder
    power_iterations : subspace iterations for the randomized range finder
    seed             : RNG seed for the randomized sketch
    """

    def __init__(self, snapshots, V, inner_product="L2", method="snapshots",
                 rank=None, oversampling=10, power_iterations=2, seed=0):
        self.snapshots = snapshots
        self.V = V
        self.mesh = V.mesh
        self.method = method
        self._ip_matrix = self._assemble_inner_product_matrix(inner_product)
        if method == "snapshots":
            self.basis, self.eigenvalues = self._compute_snapshots()
            self._total_energy = float(self.eigenvalues.sum())
        elif method == "randomized":
            if rank is None:
                raise ValueError("method='randomized' requires rank=<number of "
                                 "modes to compute>")
            self.basis, self.eigenvalues = self._compute_randomized(
                rank, oversampling=oversampling,
                power_iterations=power_iterations, seed=seed)
            self._total_energy = self._total_variance()
        else:
            raise ValueError(f"method must be 'snapshots' or 'randomized', "
                             f"got {method!r}")

    @staticmethod
    def load_snapshots(pattern):
        files = [f for f in sorted(glob(pattern))
                 if "dof_coords" not in f and "dof_cells" not in f]
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
        files = [f for f in sorted(glob(pattern))
                 if "dof_coords" not in f and "dof_cells" not in f]
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

    def _compute_snapshots(self):
        S = self.snapshots
        HS = self._ip_matrix @ S.T        # (n_dofs, n_snapshots)
        C = S @ HS                         # (n_snapshots, n_snapshots) correlation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        basis = S.T @ eigenvectors[:, idx] / np.sqrt(eigenvalues[idx])
        return basis, eigenvalues[idx]

    def _m_orthonormalize(self, Y):
        """Return Q with ``span(Q) = span(Y)`` and ``Qᵀ M Q = I`` (M = inner-product
        matrix). Whitens via an eigendecomposition of the small (ℓ×ℓ) M-Gram
        matrix, which also drops numerically null directions — so it tolerates a
        rank-deficient sketch."""
        M = self._ip_matrix
        G = Y.T @ (M @ Y)                  # (ℓ, ℓ) Gram in the M-inner product
        G = 0.5 * (G + G.T)
        w, Vg = np.linalg.eigh(G)
        keep = w > w.max() * 1e-12
        return Y @ (Vg[:, keep] / np.sqrt(w[keep]))

    def _compute_randomized(self, rank, oversampling=10, power_iterations=2, seed=0):
        """Randomized M-weighted SVD of the snapshot matrix (Halko-Martinsson-Tropp
        range finder carried out in the L²/H¹ inner product).

        Returns the top ``rank`` M-orthonormal POD modes and their eigenvalues
        without forming the (n_dof × n_snap) weighted snapshot matrix or the full
        spectrum. The modes match those of :meth:`_compute_snapshots` up to the
        truncation rank. Because the range finder M-orthonormalizes (rather than
        lumping M to a diagonal), it is exact for the consistent L² *and* H¹
        inner products.
        """
        S = self.snapshots                       # (n_snap, n_dof)
        M = self._ip_matrix
        n_snap = S.shape[0]
        ell = min(rank + oversampling, n_snap)
        rng = np.random.default_rng(seed)

        # Sketch the range of Sᵀ (it lives in the n_snap-dimensional snapshot
        # span), then run a few subspace iterations with the M-self-adjoint
        # operator K = Sᵀ S M to lock onto the dominant M-singular subspace,
        # M-orthonormalizing between iterations.
        Q = self._m_orthonormalize(S.T @ rng.standard_normal((n_snap, ell)))
        for _ in range(power_iterations):
            Q = self._m_orthonormalize(S.T @ (S @ (M @ Q)))

        # Sᵀ ≈ Q (Qᵀ M Sᵀ); the small factor is B = S M Q (n_snap × ℓ). Its right
        # singular vectors rotate Q into the POD modes; singular values² = λ.
        B = S @ (M @ Q)                          # (n_snap, ℓ)
        _, sigma, Pt = np.linalg.svd(B, full_matrices=False)
        basis = Q @ Pt.T                         # (n_dof, ℓ), M-orthonormal columns
        eigenvalues = sigma ** 2
        return basis[:, :rank], eigenvalues[:rank]

    def _total_variance(self, chunk=64):
        """Exact total POD energy ``Σλ = tr(S M Sᵀ) = Σ_i ⟨s_i, s_i⟩_M``, streamed
        in snapshot chunks so the (n_dof × chunk) intermediate stays small. Lets
        :meth:`n_modes` report a correct energy fraction under randomized
        truncation (where ``self.eigenvalues`` holds only the top ``rank``)."""
        S, M = self.snapshots, self._ip_matrix
        total = 0.0
        for a in range(0, S.shape[0], chunk):
            Sc = S[a:a + chunk]                  # (c, n_dof)
            total += float(np.einsum("ij,ji->", Sc, M @ Sc.T))
        return total

    def plot_eigenvalues(self):
        plt.semilogy(self.eigenvalues, marker="o")
        plt.xlabel("Mode")
        plt.ylabel("Eigenvalue")
        plt.title("POD Eigenvalues")
        plt.show()

    def n_modes(self, energy_fraction=0.9999):
        """Return the number of modes needed to capture *energy_fraction* of total energy."""
        cumsum = np.cumsum(self.eigenvalues)
        target = energy_fraction * self._total_energy
        if cumsum[-1] < target:
            logger.warning(
                "POD(method=%r): %d computed mode(s) capture only %.6g of the "
                "energy, short of the requested %.4g — increase rank to truncate "
                "at this threshold.", self.method, len(self.eigenvalues),
                cumsum[-1] / self._total_energy, energy_fraction)
            return len(self.eigenvalues)
        return int(np.searchsorted(cumsum, target)) + 1

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
