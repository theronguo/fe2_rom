"""
Snap-back example: deep parabolic arch under concentrated crown load.

FORCE-CONTROLLED via a Neumann traction on a narrow crown patch.
The arc-length method traces the full (crown displacement, load factor) curve.

Snap-back means that at some point BOTH λ AND crown displacement reverse
simultaneously — the fold in the equilibrium path curves back on itself so
that neither λ alone nor displacement alone is a valid continuation parameter.
This requires:
  • A deep arch (R/L ≈ 0.4) so that the inverted snap releases elastic energy.
  • A concentrated load (Neumann traction on a narrow crown patch) rather than
    a uniform body force — the concentrated load maximises snap-back severity.

Geometry
--------
Parabolic arch: span L=10, rise R=4, cross-section H×W=0.3×1.
Both ends fully pinned (u=0).
Crown patch: |x − L/2| < patch_half — top-face facets carry the traction.

Run:
    python run_snap_back.py
"""
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc

from hyperelastic_solver import (
    CylindricalArcLength,
    HyperelasticStabilitySolver,
    VTXManager,
    NeoHookean,
    setup_logging,
)

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Geometry parameters ───────────────────────────────────────────────────────
L           = 10.0   # span (m)
R           =  4.0   # arch rise at crown (m)  — R/L = 0.40 → snap-back regime
H           =  0.3   # cross-section height (m)
W           =  1.0   # cross-section width  (m)
patch_half  =  0.8   # half-width of crown load patch (m)

# ── Mesh: flat box deformed to parabolic arch ─────────────────────────────────
msh = dmesh.create_box(
    comm,
    [[0.0, 0.0, 0.0], [L, H, W]],
    [40, 4, 4],
    cell_type=dmesh.CellType.tetrahedron,
)
# Parabolic profile: shift(x) = R·(1 − (2x/L − 1)²) → 0 at ends, R at crown
msh.geometry.x[:, 1] += R * (1.0 - (2.0 * msh.geometry.x[:, 0] / L - 1.0) ** 2)

# ── Facet tags for the crown patch (top face of the box, y ≈ H) ──────────────
fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, msh.topology.dim)

def _on_crown_top(x):
    """Top face of the undeformed box within the crown patch."""
    return (np.abs(x[0] - L / 2) < patch_half) & (x[1] > H - 1e-8)

crown_facets = dmesh.locate_entities_boundary(msh, fdim, _on_crown_top)
crown_tag_id = 7
facet_indices = np.array(crown_facets, dtype=np.int32)
facet_markers = np.full(len(facet_indices), crown_tag_id, dtype=np.int32)
facet_tags = dmesh.meshtags(msh, fdim, facet_indices, facet_markers)

ds_crown = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)(crown_tag_id)

# ── Traction constant (updated by load_fn) ────────────────────────────────────
# Downward traction in reference y-direction; λ scales the magnitude.
load_intensity = 5.0   # N/m²
traction_const = fem.Constant(msh, (0.0, 0.0, 0.0))

# ── Material ─────────────────────────────────────────────────────────────────
material = NeoHookean(mu=20.0, lmbda=40.0)

solver = HyperelasticStabilitySolver(
    msh, None, None, material,
    neumann_terms=[(traction_const, ds_crown)],
    enable_viz_fields=True,
)

# ── Boundary conditions: both ends fully pinned ───────────────────────────────
zero = fem.Constant(msh, PETSc.ScalarType(0.0))
tol  = 1e-10

for comp in (0, 1, 2):
    solver.add_bc(comp, lambda x: x[0] < tol,     zero)
    solver.add_bc(comp, lambda x: x[0] > L - tol, zero)

solver.setup()

# ── Load function (force-controlled via Neumann traction) ─────────────────────
def load_fn(lam: float) -> None:
    traction_const.value = (0.0, -lam * load_intensity, 0.0)

# ── Crown displacement monitor ────────────────────────────────────────────────
dx_msh = ufl.Measure("dx", domain=msh)

def _in_crown_vol(x):
    return np.abs(x[0] - L / 2) < patch_half

crown_cells = dmesh.locate_entities(msh, msh.topology.dim, _in_crown_vol)
crown_vol_tags = dmesh.meshtags(
    msh, msh.topology.dim,
    crown_cells, np.ones(len(crown_cells), dtype=np.int32)
)
dx_crown = ufl.Measure("dx", domain=msh, subdomain_data=crown_vol_tags)(1)

crown_vol_form  = fem.form(fem.Constant(msh, 1.0) * dx_crown)
crown_disp_form = fem.form(-solver.u[1] * dx_crown)   # −u_y → positive = downward

crown_vol = comm.allreduce(fem.assemble_scalar(crown_vol_form), op=MPI.SUM)

lambdas     = [0.0]
crown_disps = [0.0]

def step_callback(lam: float) -> None:
    d = comm.allreduce(fem.assemble_scalar(crown_disp_form), op=MPI.SUM) / crown_vol
    lambdas.append(lam)
    crown_disps.append(d)
    logger.info("   crown_disp=% .4f  λ=% .4f", d, lam)

# ── Output ────────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
vtx = VTXManager(comm, "output/snap_back_solution.bp",
                 [solver.u_int, solver.F_func, solver.P_func, solver.J_func])

# ── Arc-length ────────────────────────────────────────────────────────────────
arc = CylindricalArcLength(
    arc_length=1.0,
    max_arc_steps=500,
    max_newton_iter=25,
    abs_tol=1e-5,
)

solver.run_arc_length(
    arc,
    load_fn=load_fn,
    lambda_init=0.0,
    lambda_max=1e6,          # no λ ceiling: snap-back makes λ non-monotone
    output_manager=vtx,
    step_callback=step_callback,
)

vtx.close()

# ── Plot: crown displacement vs load factor ───────────────────────────────────
if comm.rank == 0 and len(lambdas) > 1:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(crown_disps, lambdas, "r-o", markersize=4, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Crown downward displacement  (m)")
    ax.set_ylabel("Load factor  λ")
    ax.set_title("Snap-back: deep arch under concentrated crown load\n"
                 "(arc-length traces the non-monotone path — both λ and disp reverse)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("output/snap_back.png", dpi=150)
    plt.close(fig)

    np.savetxt(
        "output/snap_back.csv",
        np.column_stack([crown_disps, lambdas]),
        delimiter=",",
        header="crown_downward_disp,lambda",
        comments="",
    )
    logger.info("Saved output/snap_back.png and output/snap_back.csv")
