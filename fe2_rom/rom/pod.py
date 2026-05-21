from dolfinx import fem
from dolfinx.io import XDMFFile
from glob import glob
import numpy as np
import ufl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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
        basis = S.T @ eigenvectors[:, idx] / np.sqrt(eigenvalues[idx])
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
