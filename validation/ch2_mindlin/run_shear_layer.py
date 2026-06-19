"""Boundary shear layer problem (Kouznetsova thesis §4.4.3) with the Mindlin
second-gradient elastic law, solved by the (material-pluggable) CH2
``MacroSecondOrderSolver`` and compared to the analytical solution (eq. 4.87).

Plane-strain strip of height H (X₂), "infinite" in X₁ (modelled as a thin
column with u₂=0 on the sides). BCs (eq. 4.85): on X₂=0,H  u₁ ∈ {0, U*}, u₂=0,
F₁₂=0, F₂₂=1. The shear F₁₂(X₂) vanishes at the walls and develops a
boundary-layer profile of width ~Ẑ=√2 Z in the interior.

Analytical shear (d/dX₂ of eq. 4.87):
    F₁₂(X₂) = U*/(A Ẑ) · [ −sinh(H/Ẑ) + sinh(X₂/Ẑ) + sinh((H−X₂)/Ẑ) ],
    A = −2 + 2 cosh(H/Ẑ) − (H/Ẑ) sinh(H/Ẑ),   Ẑ = √2 Z.

Run (serial) inside the dolfinx-rve container:
    conda activate fe2_rom_env && python validation/ch2_mindlin/run_shear_layer.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, geometry
from dolfinx.mesh import create_rectangle, CellType, GhostMode

from fe2_rom.hyperelastic_solver import TimeStepper, setup_logging, broadcast_logger
from fe2_rom.ch2 import MacroSecondOrderSolver
from mindlin_material import MindlinCh2Material

# --- parameters (§4.4.3) ---
MU, KAPPA, Z = 2000.0, 5000.0, 0.05      # MPa, MPa, mm
H, USTAR = 1.0, 0.03                     # mm
# The thesis enforces X₁-uniformity with periodic BCs on the sides. We instead
# use a strip several Ẑ wide (Ẑ=√2 Z) with u₂=0 on the free sides and sample
# F₁₂ at mid-width: the free-edge perturbation decays within ~Ẑ of the sides, so
# the centreline recovers the 1-D solution. (Edge effect at mid-width: 2.4% for
# W=Z, 1.4% for W=2Z, 0.5% for W=8Z — i.e. → 0 as W grows.)
W = 8.0 * Z                              # ≈ 5.7 Ẑ wide
NX = 8                                   # elements across the width
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def F12_analytic(x2):
    Zh = np.sqrt(2.0) * Z
    A = -2.0 + 2.0 * np.cosh(H / Zh) - (H / Zh) * np.sinh(H / Zh)
    return (USTAR / (A * Zh)) * (-np.sinh(H / Zh)
                                 + np.sinh(x2 / Zh) + np.sinh((H - x2) / Zh))


def solve_one(comm, ny):
    mesh = create_rectangle(comm, [np.array([0.0, 0.0]), np.array([W, H])],
                            [NX, ny], CellType.quadrilateral,
                            ghost_mode=GhostMode.shared_facet)
    material = MindlinCh2Material(mu=MU, Z=Z, kappa=KAPPA, gdim=2)
    solver = MacroSecondOrderSolver(mesh, n_qp=2, material=material, degree=1)

    is_bot = lambda x: np.isclose(x[1], 0.0)
    is_top = lambda x: np.isclose(x[1], H)
    is_side = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], W))
    is_corner = lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    zero = fem.Constant(mesh, PETSc.ScalarType(0.0))
    utop = fem.Constant(mesh, PETSc.ScalarType(0.0))

    # u₁: 0 at bottom, U* at top
    solver.add_bc(0, is_bot, zero)
    solver.add_bc(0, is_top, utop)
    # u₂ = 0 on the whole boundary (symmetry / X₁-uniformity)
    solver.add_bc(1, is_bot, zero)
    solver.add_bc(1, is_top, zero)
    solver.add_bc(1, is_side, zero)
    # higher-order BCs (eq 4.85): F₁₂=0 (H_xy, c=1) and F₂₂=1 (H_yy, c=3) on walls
    solver.add_bc((1, 1), is_bot, zero)
    solver.add_bc((1, 1), is_top, zero)
    solver.add_bc((1, 3), is_bot, zero)
    solver.add_bc((1, 3), is_top, zero)
    # gauge: pin the remaining F̂ components (H_xx c=0, H_yx c=2) at one corner
    solver.add_bc((1, 0), is_corner, zero, pointwise=True)
    solver.add_bc((1, 2), is_corner, zero, pointwise=True)
    solver.setup()

    solver.solve(output_dir=OUT,
                 timestepper=TimeStepper(t_end=1.0, dt_init=0.25, dt_min=1e-4,
                                         dt_max=0.25, good_newton_steps=6),
                 loadhistory=lambda t: setattr(utop, "value", t * USTAR))

    # --- extract F₁₂ = H_xy along the centerline ---
    Hc = solver.w.sub(1).collapse()
    ys = np.linspace(0.0, H, 201)
    pts = np.column_stack([np.full_like(ys, W / 2), ys, np.zeros_like(ys)])
    bb = geometry.bb_tree(mesh, mesh.topology.dim)
    cand = geometry.compute_collisions_points(bb, pts)
    coll = geometry.compute_colliding_cells(mesh, cand, pts)
    cells = [coll.links(i)[0] for i in range(len(pts))]
    Hvals = Hc.eval(pts, cells)                 # (npts, 4) row-major H_ij
    return ys, Hvals[:, 1]                       # H_xy = F̂_xy = F₁₂


def main():
    comm = MPI.COMM_WORLD
    setup_logging(comm, level=logging.INFO)
    broadcast_logger("fe2_rom.ch2.macrosolver", level=logging.WARNING)
    os.makedirs(OUT, exist_ok=True)

    ny_list = [3, 5, 10, 20]
    results, errors = {}, {}
    for ny in ny_list:
        ys, fe = solve_one(comm, ny)
        an = F12_analytic(ys)
        rel = float(np.linalg.norm(fe - an) / np.linalg.norm(an))
        results[ny] = (ys, fe)
        errors[ny] = rel
        if comm.rank == 0:
            print(f"  NY={ny:3d}  (h/Ẑ={H/ny/(np.sqrt(2)*Z):.2f})  "
                  f"F₁₂ rel-L2 = {rel*100:.3f} %")

    if comm.rank == 0:
        print(f"\n==== Boundary shear layer: CH2-Mindlin vs analytical (eq 4.87) ====")
        print(f"  H={H}  Z={Z}  Ẑ={np.sqrt(2)*Z:.4f}  U*={USTAR}")
        print("  Convergence (mesh refinement) toward analytical:")
        prev = None
        for ny in ny_list:
            rate = "" if prev is None else f"  rate≈{np.log2(prev/errors[ny]):.2f}"
            print(f"    NY={ny:3d}: {errors[ny]*100:.3f} %{rate}")
            prev = errors[ny]
        _plot(results, errors, ny_list)


def _plot(results, errors, ny_list):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc})"); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ys = results[ny_list[-1]][0]
    ax[0].plot(F12_analytic(ys), ys, "-", lw=2.2, color="k", label="analytical (eq 4.87)")
    for ny in ny_list:
        y, fe = results[ny]
        ax[0].plot(fe, y, "--", lw=1.1, label=f"FE NY={ny}")
    ax[0].set_xlabel(r"shear $F_{12}$"); ax[0].set_ylabel(r"$X_2$ [mm]")
    ax[0].set_title("Boundary shear layer profile"); ax[0].legend(fontsize=7)
    ny = np.array(ny_list, float)
    ax[1].loglog(H / ny, [errors[n] * 100 for n in ny_list], "o-")
    ax[1].set_xlabel(r"element size $h = H/N_y$ [mm]")
    ax[1].set_ylabel(r"$F_{12}$ rel. L2 error [%]")
    ax[1].set_title("Convergence to analytical"); ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "shear_layer.png")
    fig.savefig(p, dpi=140); print(f"  saved {p}")


if __name__ == "__main__":
    main()
