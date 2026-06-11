"""Two-scale micromorphic FOM for van Bree et al. (2020), Example 1.

Reproduces the local-vs-global buckling column of Section 4.1:
  - Microstructure: 2ℓ × 2ℓ RVE with a square stacking of 4 circular holes
    (d = 8.67 mm, ℓ = 9.97 mm); periodic; quadratic P2 triangles, h_m = ℓ/10.
  - Hyperelastic law: ψ = c1(I1−3) + c2(I1−3)² − 2c1 lnJ + ½K(J−1)²,
    with c1 = 0.55 MPa, c2 = 0.3 MPa, K = 55 MPa.
  - One analytical patterning mode φ₁ (Eq. 56), normalised so that
    ⟨‖φ₁‖⟩_Q = 1.
  - Macroscopic specimen: W = 4ℓ × H = 8ℓ, quadratic triangles, 3 Gauss pts.
  - Compression at top/bottom edges, lateral clamping (u_x = 0 there),
    micromorphic Dirichlet v_1 = 0 at top/bottom (stiff boundary layer).
  - Output: nominal stress P_22 = (reaction at top edge) / W vs strain u/H.

Pre-requisite::
    python create_rve_mesh.py     # writes ./rve.msh

Run::
    mamba activate fe2_rom_env
    python run_macro_micromorphic.py --hM 2.0 --n_steps 50 --max_strain 0.07
"""

import os
# Keep nested RVE solves single-threaded to avoid oversubscription.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import logging

import numpy as np
from mpi4py import MPI

from dolfinx import fem
from dolfinx.mesh import create_rectangle, CellType, GhostMode

from fe2_rom.hyperelastic_solver import (
    ReactionForceLogger,
    setup_logging,
    broadcast_logger,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper
from fe2_rom.mm.material import MicromorphicRVEMaterial
from fe2_rom.mm.macrosolver import MacroMicromorphicSolver
from fe2_rom.mm.microsolver import MicroSolver
from fe2_rom.mm import set_analytical_modes

from fe2_rom.hyperelastic_solver.material import BertoldiHyperelastic

# ---------------------------------------------------------------------------
# Geometry / material constants (paper Section 4.1, Table 1)
# ---------------------------------------------------------------------------
ELL = 9.97          # mm
HOLE_D = 8.67       # mm
C1, C2, K = 0.55, 0.3, 55.0   # MPa
N_MODES = 1                   # single patterning mode (square stacking)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hM", type=float, default=2.0,
                   help="macro element size in units of ℓ (default 2.0 → Fig 4c)")
    p.add_argument("--n_steps", type=int, default=50,
                   help="number of load steps (default 50)")
    p.add_argument("--max_strain", type=float, default=0.07,
                   help="final u/H (default 0.07)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="output directory (default ./output_hM<...>)")
    p.add_argument("--verbose_rve", action="store_true",
                   help="show per-qp RVE Newton/stability output")
    p.add_argument("--objective", action="store_true",
                   help="enable the objectivity (F̄=R U) reduction in each RVE")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Analytical φ₁ (van Bree et al. Eq. 56), supplied as a coordinate→value
# callable to the generic fe2_rom.mm helper.
# ---------------------------------------------------------------------------
def phi1_mode(coords):
    """The square-stacking patterning mode φ₁ (Eq. 56), evaluated at the
    ``(n_points, 2)`` coordinate array; returns ``(n_points, 2)`` values."""
    X1, X2 = coords[:, 0], coords[:, 1]
    s_plus = np.sin(np.pi * (X1 + X2) / ELL)
    s_minus = np.sin(np.pi * (-X1 + X2) / ELL)
    return np.column_stack([-s_plus - s_minus, s_plus - s_minus])


def _populate_phi(rve, vol_global=None):
    """Write φ₁ into ``rve._phi[0]`` and renormalise (mean ‖φ₁‖ over the meshed
    cell = 1, matching the original ∫dX normalisation), then rebuild constraints.
    """
    set_analytical_modes([rve._phi[0]], phi1_mode, dx=rve.dx, comm=rve.comm,
                         normalize="mean", rve_volume=None)
    rve.rebuild_constraints()


# ---------------------------------------------------------------------------
# RVE factory (one fresh FOM per macro Gauss point).
# ---------------------------------------------------------------------------
rve_vol=(2.0 * ELL) ** 2
def make_rve_factory(rve_mesh_path: str, output_root: str,
                     objective_reduction: bool = False):
    def rve_factory(rank: int, index: int):
        out_dir = os.path.join(output_root, f"rve_{rank}_{index}")
        rve = MicroSolver(
            mesh_path=rve_mesh_path,
            comm=MPI.COMM_SELF,
            gdim=2,
            material=BertoldiHyperelastic(c1=C1, c2=C2, K=K),
            N=N_MODES,
            degree=2,
            output_dir=out_dir,
            check_stability=False,
            visualize_fields=[],
            newton_options={
                "rel_tol": 1e-8, "abs_tol": 1e-6,
                "max_iter": 30, "div_rel_tol": 10.0
            },
            timestepper_options={
                "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5,
                "dt_max": 1.0, "good_newton_steps": 5,
            },
            stability_options={
                "nev": 5,
                "neg_tol": -1e-8,
                "petsc_options": {
                    "st_ksp_type": "preonly",
                    "st_pc_type": "lu",
                },
            },
            averages_only_final=True,
            rve_volume=rve_vol,    # |Q| of the square periodic cell
            # Objectivity (F̄ = R U) reduction; off by default → unchanged.
            objective_reduction=objective_reduction,
        )
        _populate_phi(rve, rve_vol)
        return rve
    return rve_factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    rve_mesh = os.path.join(HERE, "rve.msh")
    if not os.path.exists(rve_mesh):
        raise FileNotFoundError(
            f"RVE mesh not found at {rve_mesh}."
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(HERE, f"output_hM{args.hM:g}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Logging (mirrors examples/mm/example_3 pattern)
    # -----------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    setup_logging(comm, level=logging.INFO)
    log = logging.getLogger("validation.mm_2d.example_1")
    log.addFilter(lambda _: comm.rank == 0)

    _RVE_LOGGERS = (
        "fe2_rom.hyperelastic_solver.solver",
        "fe2_rom.hyperelastic_solver.solvers",
        "fe2_rom.hyperelastic_solver.stability",
        "fe2_rom.ch1.microsolver",
        "fe2_rom.mm.microsolver",
        "fe2_rom.rom.solver_ch1",
        "fe2_rom.rom.solver_mm",
    )
    if args.verbose_rve:
        broadcast_logger(*_RVE_LOGGERS, level=logging.DEBUG)
    else:
        for name in _RVE_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

    log.info("=" * 70)
    log.info("van Bree (2020) — Example 1: local-vs-global buckling")
    log.info("ℓ = %.3f mm   d = %.3f mm   W = %.3f mm   H = %.3f mm",
             ELL, HOLE_D, 6 * ELL, 30 * ELL)
    log.info("hM = %g·ℓ = %.3f mm     n_steps = %d     max(u/H) = %g",
             args.hM, args.hM * ELL, args.n_steps, args.max_strain)
    log.info("Material (Eq. 55):  c1=%.2f  c2=%.2f  K=%.2f  MPa", C1, C2, K)
    log.info("=" * 70)

    # -----------------------------------------------------------------------
    # Macro mesh: rectangle [0, W] × [0, H], structured triangles
    # -----------------------------------------------------------------------
    W = 6.0 * ELL
    H = 30.0 * ELL
    h_M = args.hM * ELL
    Nx = max(1, int(round(W / h_M)))
    Ny = max(1, int(round(H / h_M)))
    log.info("Macro mesh: %d × %d quad-pairs (=> %d triangles)", Nx, Ny, 2 * Nx * Ny)

    macro_mesh = create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([W, H])],
        [Nx, Ny],
        CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    # -----------------------------------------------------------------------
    # Constitutive law + macro solver
    # -----------------------------------------------------------------------
    rve_factory = make_rve_factory(
        rve_mesh_path=rve_mesh,
        output_root=os.path.join(args.output_dir, "rve_workdirs"),
        objective_reduction=args.objective,
    )
    if args.objective:
        log.info("Objectivity reduction ENABLED (drive RVEs with U, symmetric adjoints)")
    material = MicromorphicRVEMaterial(rve_factory, N_modes=N_MODES, gdim=2)

    solver = MacroMicromorphicSolver(
        macro_mesh,
        n_qp=2,                  # 3 Gauss pts / triangle (verified via basix)
        N_modes=N_MODES,
        material=material,
        degree=2,                # quadratic macro triangles (paper Section 4.1)
        check_stability=True,
        enable_restart=True
    )

    # -----------------------------------------------------------------------
    # Boundary conditions
    # -----------------------------------------------------------------------
    bot_disp = fem.Constant(macro_mesh, 0.0)     # +u/2 at bottom
    top_disp = fem.Constant(macro_mesh, 0.0)     # −u/2 at top (measured)
    zero     = fem.Constant(macro_mesh, 0.0)

    is_bottom = lambda x: np.isclose(x[1], 0.0)
    is_top    = lambda x: np.isclose(x[1], H)

    # u_x = 0 on top and bottom (clamped lateral motion)
    solver.add_bc((0, 0), is_bottom, zero)
    solver.add_bc((0, 0), is_top,    zero)
    # u_y = ±u/2 on bottom / top; measure reaction at the top
    solver.add_bc((0, 1), is_bottom, bot_disp)
    solver.add_bc((0, 1), is_top,    top_disp, measure_reaction=True)
    # v_1 = 0 at the loaded edges (stiff boundary layer)
    solver.add_bc((1,), is_bottom, zero)
    solver.add_bc((1,), is_top,    zero)

    solver.setup()

    # -----------------------------------------------------------------------
    # Load ramp:  u/H(t) = t · max_strain;  u(t) = t · max_strain · H
    # -----------------------------------------------------------------------
    def loadhistory(t: float) -> None:
        u_total = t * args.max_strain * H
        bot_disp.value = +0.5 * u_total
        top_disp.value = -0.5 * u_total

    timestepper = TimeStepper(
        t_end=1.0,
        dt_init=1.0 / args.n_steps,
        dt_min=1e-5,
        dt_max=1.0 / args.n_steps,
        good_newton_steps=5,
    )

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    reaction_logger = ReactionForceLogger()
    solver.solve(
        output_dir=args.output_dir,
        timestepper=timestepper,
        loadhistory=loadhistory,
        reaction_logger=reaction_logger,
        save_macro_history=True,
        vtx_segment_per_resume=True,
        # rve_history_qps=[(0, 0)],
        save_qp_history=True
    )
    reaction_logger.save(
        comm,
        os.path.join(args.output_dir, "reaction.png"),
        os.path.join(args.output_dir, "reaction.csv"),
    )

    # -----------------------------------------------------------------------
    # Post-process to P_22 vs u/H (rank-0 only)
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        disp = np.asarray(reaction_logger.displacements, dtype=float)
        rxn  = np.asarray(reaction_logger.forces, dtype=float)
        # disp = top_disp.value = -u/2  →  u/H = -2 disp / H
        uH = -2.0 * disp / H
        # Top reaction is negative (material pushes up against compression);
        # nominal compressive stress P_22 = -R / W  (positive in compression).
        P22_MPa = -rxn / W
        P22_kPa = 1000.0 * P22_MPa

        csv_path = os.path.join(args.output_dir, "p22_vs_uH.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["u_over_H", "P22_MPa", "P22_kPa"])
            for a, b, c in zip(uH, P22_MPa, P22_kPa):
                w.writerow([f"{a:.8e}", f"{b:.8e}", f"{c:.8e}"])
        log.info("Wrote %s", csv_path)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            # prepend (0, 0) to show the initial linear regime; skip the first point if it's already (0, 0)
            uH = np.insert(uH, 0, 0.0) if uH[0] != 0.0 else uH
            P22_kPa = np.insert(P22_kPa, 0, 0.0) if P22_kPa[0] != 0.0 else P22_kPa
            ax.plot(uH, P22_kPa, "-o", ms=3)
            ax.set_xlabel(r"$u/H$ [-]")
            ax.set_ylabel(r"$P_{22}$ [kPa]")
            ax.set_title(f"van Bree Ex.1, $h_M = {args.hM}\\,\\ell$")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png_path = os.path.join(args.output_dir, "p22_vs_uH.png")
            fig.savefig(png_path, dpi=150)
            log.info("Wrote %s", png_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Plot skipped: %s", exc)

    log.info("Done.  Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
