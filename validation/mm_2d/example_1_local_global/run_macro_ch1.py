"""First-order (classical) FE² driver for van Bree (2020), Example 1.

Companion to ``run_macro_micromorphic.py``: same macro geometry (W = 4ℓ,
H = 8ℓ), same RVE (``rve.msh`` — 2ℓ × 2ℓ, four holes, P2 triangles, h_m = ℓ/10),
same hyperelastic law, same BCs — but the macroscopic continuum is the
classical first-order homogenization (no patterning enrichment fields).
This serves as the baseline against which the micromorphic FE² is compared
(cf. paper §4.1: first-order FE² is known to deviate by up to 40% from DNS
in the post-buckling regime).

Pre-requisite::
    python create_rve_mesh.py     # writes ./rve.msh

Run::
    mamba activate fe2_rom_env
    python run_macro_ch1.py --hM 2.0 --n_steps 50 --max_strain 0.07
    mpirun -n 4 python run_macro_ch1.py ...
"""

import os
# Keep nested RVE solves single-threaded.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import logging

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_rectangle, CellType, GhostMode

from fe2_rom.hyperelastic_solver import (
    ReactionForceLogger,
    TimeStepper,
    setup_logging,
    broadcast_logger,
)
from fe2_rom.ch1 import MacroSolver

from material import BertoldiHyperelastic

# ---------------------------------------------------------------------------
# Geometry / material constants (paper Section 4.1, Table 1)
# ---------------------------------------------------------------------------
ELL = 9.97
HOLE_D = 8.67
C1, C2, K = 0.55, 0.3, 55.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hM", type=float, default=2.0,
                   help="macro element size in units of ℓ (default 2.0)")
    p.add_argument("--n_steps", type=int, default=50,
                   help="number of load steps (default 50)")
    p.add_argument("--max_strain", type=float, default=0.07,
                   help="final u/H (default 0.07)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="output directory (default ./output_ch1_hM<...>)")
    p.add_argument("--verbose_rve", action="store_true",
                   help="show per-qp RVE Newton/stability output")
    return p.parse_args()


def main():
    args = parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    rve_mesh = os.path.join(HERE, "rve.msh")
    if not os.path.exists(rve_mesh):
        raise FileNotFoundError(
            f"RVE mesh not found at {rve_mesh}. Run `python create_rve_mesh.py` first."
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(HERE, f"output_ch1_hM{args.hM:g}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Logging (same pattern as the mm driver)
    # -----------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    setup_logging(comm, level=logging.INFO)
    log = logging.getLogger("validation.mm_2d.example_1.ch1")
    log.addFilter(lambda _: comm.rank == 0)

    _RVE_LOGGERS = (
        "fe2_rom.hyperelastic_solver.solver",
        "fe2_rom.hyperelastic_solver.solvers",
        "fe2_rom.hyperelastic_solver.stability",
        "fe2_rom.ch1.microsolver",
        "fe2_rom.rom.solver_ch1",
    )
    if args.verbose_rve:
        broadcast_logger(*_RVE_LOGGERS, level=logging.DEBUG)
    else:
        for name in _RVE_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

    log.info("=" * 70)
    log.info("van Bree (2020) — Example 1 (CH1, first-order FE²)")
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
        ghost_mode=GhostMode.none,
    )

    # -----------------------------------------------------------------------
    # FE² macro solver — full-order RVE per macro qp.
    # -----------------------------------------------------------------------
    solver = MacroSolver(
        mesh=macro_mesh,
        full=True,
        n_qp=2,                             # 3 Gauss pts / triangle
        rve_mesh_path=rve_mesh,
        rve_material=BertoldiHyperelastic(c1=C1, c2=C2, K=K),
        gdim=2,
        rve_degree=2,                       # P2 RVE (matches mesh order)
        rve_output_dir=os.path.join(args.output_dir, "rve_workdirs"),
        rve_visualize_fields=[],
        rve_average_quantities=["P", "A"],  # → Pbar, dPbar_dFbar
        rve_check_stability=True,
        rve_newton_options={
            "rel_tol": 1e-8, "abs_tol": 1e-6,
            "max_iter": 50, "div_rel_tol": 10.0,
            "switch_to_minres": True,
        },
        rve_timestepper_options={
            "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5,
            "dt_max": 1.0, "good_newton_steps": 5,
        },
        rve_averages_only_final=True,
        rve_volume=(2.0 * ELL) ** 2,        # |Q| of the square periodic cell
        degree=2,                           # quadratic macro triangles
        check_stability=True,
    )

    # -----------------------------------------------------------------------
    # Boundary conditions (same as the mm driver):
    #   u_x = 0 on top and bottom (clamped lateral motion),
    #   u_y = +u/2 on bottom, u_y = -u/2 on top (top measures the reaction).
    # -----------------------------------------------------------------------
    bot_disp = fem.Constant(macro_mesh, PETSc.ScalarType(0.0))
    top_disp = fem.Constant(macro_mesh, PETSc.ScalarType(0.0))
    zero     = fem.Constant(macro_mesh, PETSc.ScalarType(0.0))

    is_bottom = lambda x: np.isclose(x[1], 0.0)
    is_top    = lambda x: np.isclose(x[1], H)

    solver.add_bc(0, is_bottom, zero)
    solver.add_bc(0, is_top,    zero)
    solver.add_bc(1, is_bottom, bot_disp)
    solver.add_bc(1, is_top,    top_disp,
                  measure_reaction=True, reaction_direction=(0.0, 1.0))

    solver.setup()

    # -----------------------------------------------------------------------
    # Load history and timestepper
    # -----------------------------------------------------------------------
    def loadhistory(t: float) -> None:
        u_total = t * args.max_strain * H
        bot_disp.value = +0.5 * u_total
        top_disp.value = -0.5 * u_total

    timestepper = TimeStepper(
        t_end=1.0,
        dt_init=1.0 / args.n_steps,
        dt_min=1e-6,
        dt_max=1.0 / args.n_steps,
        good_newton_steps=5,
    )

    reaction_logger = ReactionForceLogger()

    solver.solve(
        output_dir=args.output_dir,
        timestepper=timestepper,
        loadhistory=loadhistory,
        output_variables=[solver.u],
        reaction_logger=reaction_logger,
        pert_amplitude_init=1e-2,
    )

    reaction_logger.save(
        comm,
        os.path.join(args.output_dir, "reaction.png"),
        os.path.join(args.output_dir, "reaction.csv"),
    )

    # -----------------------------------------------------------------------
    # Post-process to P_22 vs u/H (rank 0)
    # -----------------------------------------------------------------------
    if comm.rank == 0:
        disp = np.asarray(reaction_logger.displacements, dtype=float)
        rxn  = np.asarray(reaction_logger.forces, dtype=float)
        uH = -2.0 * disp / H
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
            ax.plot(uH, P22_kPa, "-o", ms=3)
            ax.set_xlabel(r"$u/H$ [-]")
            ax.set_ylabel(r"$P_{22}$ [kPa]")
            ax.set_title(f"CH1 FE², $h_M = {args.hM}\\,\\ell$")
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
