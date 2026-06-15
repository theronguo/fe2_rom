"""Second-order (CH2) FE² driver for van Bree (2020), Example 1.

Companion to ``run_macro_ch1.py`` / ``run_macro_mm.py``: same macro geometry
(W = 6ℓ, H = 30ℓ), same RVE (``rve.msh`` — 2ℓ × 2ℓ, four holes, P2 triangles,
centred at the origin), same hyperelastic law and BCs — but the macroscopic
continuum is the second-order (strain-gradient) homogenization of
Guo et al. (2024), with the mixed ``[u, F̂, L̄]`` formulation (paper Eqs. 3-6).

The macro mesh is triangular with the inf-sup-stable P2-P1-P0 (u-F̂-L̄) element
— the same discretization as the other runners and the CH2 ROM runner, so the
resulting P₂₂–u/H curves compare apples-to-apples.

Pre-requisite::
    python create_dns_mesh.py --nx 2 --ny 2 --center --output rve.msh

Run::
    mamba activate fe2_rom_env
    python run_macro_ch2.py --hM 2.0 --n_steps 50 --max_strain 0.07
    mpirun -n 4 python run_macro_ch2.py ...
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

from dolfinx import fem
from dolfinx.mesh import create_rectangle, CellType, GhostMode

from fe2_rom.hyperelastic_solver import (
    ReactionForceLogger,
    TimeStepper,
    setup_logging,
    broadcast_logger,
)
from fe2_rom.hyperelastic_solver.material import BertoldiHyperelastic
from fe2_rom.ch2 import MacroSecondOrderSolver, Ch2RVEMaterial
from fe2_rom.ch2.microsolver import MicroSolver

# ---------------------------------------------------------------------------
# Geometry / material constants (paper Section 4.1, Table 1)
# ---------------------------------------------------------------------------
ELL = 9.97
HOLE_D = 8.67
C1, C2, K = 0.55, 0.3, 55.0
RVE_VOL = (2.0 * ELL) ** 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hM", type=float, default=2.0,
                   help="macro element size in units of ℓ (default 2.0)")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--max_strain", type=float, default=0.07, help="final u/H")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--lagrange_stab", type=float, default=0.0,
                   help="DG0 multiplier regularization ε (1/stress); only needed "
                        "for the unstable Q1–Q1–Q0 element. Default 0 (the "
                        "Q2–Q1–Q0 element is inf-sup stable).")
    p.add_argument("--rve_pert_amplitude", type=float, default=1e-2,
                   help="initial eigenmode-kick amplitude for each RVE's "
                        "post-buckling traversal (thin-strut RVEs may need ~1e-3).")
    p.add_argument("--macro_check_stability", action="store_true",
                   help="enable macro saddle-inertia stability (perturb at "
                        "n_neg=m+1, reject at n_neg>=m+2).")
    p.add_argument("--macro_pert_amplitude", type=float, default=1e-2,
                   help="initial eigenmode-kick amplitude for the macro "
                        "buckling-mode perturbation.")
    p.add_argument("--verbose_rve", action="store_true")
    return p.parse_args()


def make_rve_factory(rve_mesh, output_root, pert_amplitude_init=1e-2):
    def rve_factory(rank, index):
        return MicroSolver(
            mesh_path=rve_mesh, comm=MPI.COMM_SELF, gdim=2,
            material=BertoldiHyperelastic(c1=C1, c2=C2, K=K),
            degree=2, output_dir=os.path.join(output_root, f"rve_{rank}_{index}"),
            check_stability=True, pert_amplitude_init=pert_amplitude_init,
            visualize_fields=[], averages_only_final=True,
            newton_options={"rel_tol": 1e-10, "abs_tol": 1e-8,
                            "max_iter": 30, "div_rel_tol": 10.0},
            timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-6,
                                 "dt_max": 1.0, "good_newton_steps": 5},
            stability_options={"nev": 5, "neg_tol": -1e-8,
                               "petsc_options": {"st_ksp_type": "preonly",
                                                 "st_pc_type": "lu"}},
            rve_volume=RVE_VOL,
        )
    return rve_factory


def main():
    args = parse_args()
    HERE = os.path.dirname(os.path.abspath(__file__))
    rve_mesh = os.path.join(HERE, "rve.msh")
    if not os.path.exists(rve_mesh):
        raise FileNotFoundError(
            f"RVE mesh not found at {rve_mesh}. Run "
            "`python create_dns_mesh.py --nx 2 --ny 2 --center --output rve.msh`.")
    if args.output_dir is None:
        args.output_dir = os.path.join(HERE, f"output_ch2_hM{args.hM:g}")
    os.makedirs(args.output_dir, exist_ok=True)

    comm = MPI.COMM_WORLD
    setup_logging(comm, level=logging.INFO)
    log = logging.getLogger("validation.mm_2d.example_1.ch2")
    log.addFilter(lambda _: comm.rank == 0)
    _RVE_LOGGERS = (
        "fe2_rom.hyperelastic_solver.solver", "fe2_rom.hyperelastic_solver.solvers",
        "fe2_rom.hyperelastic_solver.stability", "fe2_rom.ch1.microsolver",
        "fe2_rom.ch2.microsolver",
    )
    if args.verbose_rve:
        broadcast_logger(*_RVE_LOGGERS, level=logging.DEBUG)
    else:
        for name in _RVE_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

    log.info("=" * 70)
    log.info("van Bree (2020) — Example 1 (CH2, second-order FE²)")
    log.info("hM = %g·ℓ     n_steps = %d     max(u/H) = %g     ε_stab = %g",
             args.hM, args.n_steps, args.max_strain, args.lagrange_stab)
    log.info("=" * 70)

    # Macro mesh: triangles on [0, W] × [0, H] with the P2-P1-P0 (u-F̂-L̄)
    # element — same discretization as the ROM runner (rom_ch2/), so a FOM-vs-ROM
    # comparison at a given hM is apples-to-apples.
    W = 6.0 * ELL
    H = 30.0 * ELL
    h_M = args.hM * ELL
    Nx = max(1, int(round(W / h_M)))
    Ny = max(1, int(round(H / h_M)))
    log.info("Macro mesh: %d × %d quad-pairs (=> %d triangles, P2-P1-P0)",
             Nx, Ny, 2 * Nx * Ny)
    macro_mesh = create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([W, H])], [Nx, Ny],
        CellType.triangle, ghost_mode=GhostMode.shared_facet)

    rve_factory = make_rve_factory(
        rve_mesh, os.path.join(args.output_dir, "rve_workdirs"),
        pert_amplitude_init=args.rve_pert_amplitude)
    material = Ch2RVEMaterial(rve_factory, gdim=2)
    solver = MacroSecondOrderSolver(
        macro_mesh, n_qp=2, material=material, degree=1,
        lagrange_stab=args.lagrange_stab,
        check_stability=args.macro_check_stability,
        enable_restart=True)

    # BCs (mirroring run_macro_ch1.py + the F̂ conditions of paper §4.1):
    #   u_x = 0 top/bottom; u_y = ±u/2 (reaction at top).
    #   F̄_xx = 1, F̄_yx = 0 on top/bottom  ⇒  H_xx = H_yx = 0 there
    #     (flattened H components: 0=xx, 1=xy, 2=yx, 3=yy).
    #   F̂ = I at the bottom-left point (pin all H components) — removes the
    #   F̂_xy / F̂_yy zero-energy modes.
    bot = fem.Constant(macro_mesh, 0.0)
    top = fem.Constant(macro_mesh, 0.0)
    zero = fem.Constant(macro_mesh, 0.0)
    is_bot = lambda x: np.isclose(x[1], 0.0)
    is_top = lambda x: np.isclose(x[1], H)
    is_corner = lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

    solver.add_bc(0, is_bot, zero)
    solver.add_bc(0, is_top, zero)
    solver.add_bc(1, is_bot, bot)
    solver.add_bc(1, is_top, top, measure_reaction=True)
    solver.add_bc((1, 0), is_bot, zero)
    solver.add_bc((1, 0), is_top, zero)
    solver.add_bc((1, 2), is_bot, zero)
    solver.add_bc((1, 2), is_top, zero)
    for c in range(4):
        solver.add_bc((1, c), is_corner, zero, pointwise=True)
    solver.setup()

    def loadhistory(t):
        u_total = t * args.max_strain * H
        bot.value = +0.5 * u_total
        top.value = -0.5 * u_total

    timestepper = TimeStepper(
        t_end=1.0, dt_init=1.0 / args.n_steps, dt_min=1e-6,
        dt_max=1.0 / args.n_steps, good_newton_steps=5)

    reaction_logger = ReactionForceLogger()
    solver.solve(output_dir=args.output_dir, timestepper=timestepper,
                 loadhistory=loadhistory, reaction_logger=reaction_logger,
                 pert_amplitude_init=args.macro_pert_amplitude)
    reaction_logger.save(
        comm, os.path.join(args.output_dir, "reaction.png"),
        os.path.join(args.output_dir, "reaction.csv"))

    if comm.rank == 0:
        disp = np.asarray(reaction_logger.displacements, dtype=float)
        rxn = np.asarray(reaction_logger.forces, dtype=float)
        uH = -2.0 * disp / H
        P22_kPa = 1000.0 * (-rxn / W)
        csv_path = os.path.join(args.output_dir, "p22_vs_uH.csv")
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["u_over_H", "P22_MPa", "P22_kPa"])
            for a, b in zip(uH, P22_kPa):
                wr.writerow([f"{a:.8e}", f"{b/1000.0:.8e}", f"{b:.8e}"])
        log.info("Wrote %s", csv_path)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            uHp = np.insert(uH, 0, 0.0) if uH.size and uH[0] != 0.0 else uH
            P22p = np.insert(P22_kPa, 0, 0.0) if P22_kPa.size and P22_kPa[0] != 0.0 else P22_kPa
            ax.plot(uHp, P22p, "-o", ms=3)
            ax.set_xlabel(r"$u/H$ [-]"); ax.set_ylabel(r"$P_{22}$ [kPa]")
            ax.set_title(f"CH2 FE², $h_M = {args.hM}\\,\\ell$")
            ax.grid(True, alpha=0.3); fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "p22_vs_uH.png"), dpi=150)
        except Exception as exc:  # noqa: BLE001
            log.warning("Plot skipped: %s", exc)
    log.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
