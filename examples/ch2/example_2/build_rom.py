"""Build the POD + ECM reduced-order model for the second-order (CH2) RVE.

Reads the ``(F̄, Ḡ)`` snapshot pool written by ``generate_training_data.py``
(``output_gen/snapshots_pool/``) and writes the ECM artefacts (bases + magic
points) to ``ecm/``.

Three POD bases feed the empirical-cubature (ECM) magic-point selection:

  * ``u_fluc`` — displacement fluctuation (H¹),
  * ``P``      — first Piola stress (L²),
  * ``Q``      — the **double- / couple-stress density**
                 ``𝒴_iJK = ½(X_K P_iJ + X_J P_iK)`` (L²),

whose volume average is the effective double stress ``Q̄`` (Eq. 27). Like the
micromorphic ``Λ_i`` block in ``examples/mm/example_1/build_rom.py``, ``𝒴`` is a
closed-form function of ``P`` and the coordinates, so it is reconstructed from
the ``P`` snapshots here and handed to the ECM as an extra volume-integral
constraint block. Including it makes the magic-point rule integrate the
higher-order stress accurately, not just ``P̄``.

Run (serial only):
    python build_rom.py
"""
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
snapshot_dir = "output_gen"
pool_dir = f"{snapshot_dir}/snapshots_pool"
mesh_file = "rve.msh"

# Set FULL_QUADRATURE=True to skip the ECM greedy and instead write a reduced
# model that uses the *full* (exact) quadrature — every point with its exact
# weight. The online solve then has no cubature error, only POD (Galerkin)
# error, so comparing `ecm_full/` against `ecm/` isolates POD vs ECM error.
# It writes to a separate dir so both models coexist; point
# run_homogenization_rom.py at whichever you want to test.
FULL_QUADRATURE = False
ecm_dir = "ecm_full" if FULL_QUADRATURE else "ecm"

ecm_tol = 1e-4
ratio_uP = 1.0
ratio_P = 1.0
ratio_Q = 1.0
energy_tol = 0.999999

mesh = io.gmsh.read_from_msh(f"{mesh_file}", comm, 0, gdim=gdim).mesh
V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
S = fem.functionspace(mesh, ("DG", 1, (gdim, gdim)))


def load_pool_snapshots(field: str, V_space):
    """Load all ``{field}_s*_*.npy`` snapshots from the merged pool and align
    DOFs to ``V_space``'s serial ordering using the shared
    ``{field}_dof_coords.npy`` written by the sampler."""
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


# --- u_fluc POD (H1) --------------------------------------------------------
snapshots_u = load_pool_snapshots("u_fluc", V)
print(f"[snapshots] u_fluc: {snapshots_u.shape}")
pod_u = POD(snapshots_u, V, inner_product="H1")
N = pod_u.n_modes(energy_tol)

u_proj = snapshots_u @ pod_u._ip_matrix @ pod_u.basis[:, :N] @ pod_u.basis[:, :N].T
h1_err = np.sqrt(np.diagonal((snapshots_u - u_proj) @ pod_u._ip_matrix @ (snapshots_u - u_proj).T))
h1_norm = np.sqrt(np.diagonal(snapshots_u @ pod_u._ip_matrix @ snapshots_u.T))
print(f"u_fluc reconstruction error: max={np.max(h1_err / h1_norm):.2%}, "
      f"mean={np.mean(h1_err / h1_norm):.2%}")

# --- P POD (L2) -------------------------------------------------------------
snapshots_P = load_pool_snapshots("P", S)
print(f"[snapshots] P:      {snapshots_P.shape}")
pod_P = POD(snapshots_P, S, inner_product="L2")
M = pod_P.n_modes(energy_tol)

P_proj = snapshots_P @ pod_P._ip_matrix @ pod_P.basis[:, :M] @ pod_P.basis[:, :M].T
l2_err = np.sqrt(np.diagonal((snapshots_P - P_proj) @ pod_P._ip_matrix @ (snapshots_P - P_proj).T))
l2_norm = np.sqrt(np.diagonal(snapshots_P @ pod_P._ip_matrix @ snapshots_P.T))
print(f"P reconstruction error: max={np.max(l2_err / l2_norm):.2%}, "
      f"mean={np.mean(l2_err / l2_norm):.2%}")

# --- Q (double-/couple-stress density) POD (L2) -----------------------------
# 𝒴_iJK = ½ (X_K P_iJ + X_J P_iK); ⟨𝒴⟩ = Q̄ (Eq. 27). Reconstructed from the
# P snapshots so no extra field needs saving in generate_training_data.py.
S_Q = fem.functionspace(mesh, ("DG", 1, (gdim, gdim, gdim)))
P_func = fem.Function(S, name="P_snap")
X = ufl.SpatialCoordinate(mesh)
Y_expr = fem.Expression(
    ufl.as_tensor([
        [[0.5 * (X[K] * P_func[i, J] + X[J] * P_func[i, K])
          for K in range(gdim)]
         for J in range(gdim)]
        for i in range(gdim)
    ]),
    S_Q.element.interpolation_points,
)
Q_func = fem.Function(S_Q)
snapshots_Q = np.zeros((snapshots_P.shape[0], Q_func.x.array.size))
for t in range(snapshots_P.shape[0]):
    P_func.x.array[:] = snapshots_P[t]
    P_func.x.scatter_forward()
    Q_func.interpolate(Y_expr)
    snapshots_Q[t] = Q_func.x.array
print(f"[snapshots] Q:      {snapshots_Q.shape}")
pod_Q = POD(snapshots_Q, S_Q, inner_product="L2")
M_Q = pod_Q.n_modes(energy_tol)

Q_proj = snapshots_Q @ pod_Q._ip_matrix @ pod_Q.basis[:, :M_Q] @ pod_Q.basis[:, :M_Q].T
lq_err = np.sqrt(np.diagonal((snapshots_Q - Q_proj) @ pod_Q._ip_matrix @ (snapshots_Q - Q_proj).T))
lq_norm = np.sqrt(np.diagonal(snapshots_Q @ pod_Q._ip_matrix @ snapshots_Q.T))
print(f"Q reconstruction error: max={np.max(lq_err / lq_norm):.2%}, "
      f"mean={np.mean(lq_err / lq_norm):.2%}")


# --- ECM: u·P + ∫P + double-stress density Q --------------------------------
ecm_kwargs = {
    "Q": {
        "basis": pod_Q.basis[:, :M_Q],
        "space": S_Q,
        "sigma": np.sqrt(pod_Q.eigenvalues[:M_Q]),
        "ratio": ratio_Q,
    }
}

ecm = ECM(
    pod_u.basis[:, :N], pod_P.basis[:, :M], V, S,
    degree=degree,
    sigma_u=np.sqrt(pod_u.eigenvalues[:N]),
    sigma_P=np.sqrt(pod_P.eigenvalues[:M]),
    ratio_uP=ratio_uP, ratio_P=ratio_P,
    compress_uP="auto",
    quad_degree=degree + 2,   # triangle6 geometry: over-integrate the tabulation
    kwargs=ecm_kwargs,
)
if FULL_QUADRATURE:
    ecm.use_full_quadrature()   # exact cubature — no hyper-reduction (POD error only)
else:
    ecm.compute_magic(tol=ecm_tol)

print(f"Energy criterion ({energy_tol:.4%}): N={N} u-modes, M={M} P-modes, "
      f"M_Q={M_Q} Q-modes")
print(f"Verified full-system residual: {ecm.true_residual:.3e}")
# (ECM prints its coverage — magic points vs total quadrature points / elements —
#  automatically after compute_magic / use_full_quadrature.)

ecm.save_variant2(f"{ecm_dir}")
