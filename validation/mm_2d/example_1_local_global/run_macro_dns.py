"""Direct Numerical Simulation (DNS) driver for van Bree (2020), Section 4.1.

Drives ``HyperelasticStabilitySolver`` on a fully-resolved perforated specimen
under prescribed-strain compression. Reproduces a single realisation of
Fig. 5(a) (default 6ℓ × 30ℓ) and produces a P_22 vs u/H curve directly
comparable to the micromorphic FE² output of ``run_macro_micromorphic.py``.

Pre-requisite::
    python create_dns_mesh.py                         # writes dns_6x30.msh
    python create_dns_mesh.py --w 4 --h 8             # for a 4ℓ × 8ℓ specimen

Run::
    mamba activate fe2_rom_env
    python run_dns.py --mesh dns_6x30.msh --max_strain 0.07
    mpirun -n 4 python run_dns.py --mesh dns_6x30.msh --max_strain 0.07
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import logging
import re

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io

from fe2_rom.hyperelastic_solver import (
    HyperelasticStabilitySolver,
    ReactionForceLogger,
    TimeStepper,
    VTXManager,
    setup_logging,
)

from fe2_rom.hyperelastic_solver.material import BertoldiHyperelastic

# ---------------------------------------------------------------------------
# Constants (paper Section 4.1, Table 1)
# ---------------------------------------------------------------------------
ELL = 9.97              # mm
C1, C2, K = 0.55, 0.3, 55.0    # MPa


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mesh", type=str, default="dns_6x30.msh",
                   help="DNS mesh file (default dns_6x30.msh)")
    p.add_argument("--n_steps", type=int, default=100,
                   help="number of load steps (default 100)")
    p.add_argument("--max_strain", type=float, default=0.05,
                   help="final compressive u/H (default 0.05)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="output directory (default ./output_dns_<wxh>)")
    p.add_argument("--no_stability", action="store_true",
                   help="disable on-the-fly stability check")
    return p.parse_args()


def main():
    args = parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    mesh_path = args.mesh if os.path.isabs(args.mesh) else os.path.join(HERE, args.mesh)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"DNS mesh not found at {mesh_path}. "
            f"Run `python create_dns_mesh.py` first."
        )

    # Derive a label from the file name (e.g. dns_6x30.msh → 6x30) for output.
    base = os.path.splitext(os.path.basename(mesh_path))[0]
    m = re.search(r"(\d+)x(\d+)", base)
    label = m.group(0) if m else base
    if args.output_dir is None:
        args.output_dir = os.path.join(HERE, f"output_dns_{label}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    comm = MPI.COMM_WORLD
    setup_logging(comm, level=logging.INFO)
    log = logging.getLogger("validation.mm_2d.example_1.dns")
    log.addFilter(lambda _: comm.rank == 0)

    # -----------------------------------------------------------------------
    # Read mesh
    # -----------------------------------------------------------------------
    mesh, cell_tags, facet_tags, _, _, _ = io.gmsh.read_from_msh(
        mesh_path, comm, 0, gdim=2
    )
    X = mesh.geometry.x
    x_min, x_max = comm.allreduce(X[:, 0].min(), op=MPI.MIN), comm.allreduce(X[:, 0].max(), op=MPI.MAX)
    y_min, y_max = comm.allreduce(X[:, 1].min(), op=MPI.MIN), comm.allreduce(X[:, 1].max(), op=MPI.MAX)
    W = x_max - x_min
    H = y_max - y_min

    log.info("=" * 70)
    log.info("van Bree (2020) — Example 1 DNS (mesh: %s)", os.path.basename(mesh_path))
    log.info("W = %.3f mm (%.2f·ℓ)   H = %.3f mm (%.2f·ℓ)   H/W = %.3f",
             W, W / ELL, H, H / ELL, H / W)
    log.info("n_steps = %d   max(u/H) = %g   stability = %s",
             args.n_steps, args.max_strain, "off" if args.no_stability else "on")
    log.info("Material (Eq. 55):  c1=%.2f  c2=%.2f  K=%.2f  MPa", C1, C2, K)
    log.info("=" * 70)

    # -----------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------
    material = BertoldiHyperelastic(c1=C1, c2=C2, K=K)
    solver = HyperelasticStabilitySolver(
        mesh, cell_tags, facet_tags, material,
        degree=2,                       # quadratic triangles (matches the mesh)
        enable_viz_fields=True,
    )

    # -----------------------------------------------------------------------
    # Boundary conditions (same as the macro problem):
    #   u_x = 0 on top and bottom,
    #   u_y = +u/2 on bottom, u_y = -u/2 on top (top measures the reaction).
    # -----------------------------------------------------------------------
    bot_disp = fem.Constant(mesh, PETSc.ScalarType(0.0))
    top_disp = fem.Constant(mesh, PETSc.ScalarType(0.0))
    zero     = fem.Constant(mesh, PETSc.ScalarType(0.0))

    is_bottom = lambda x: np.isclose(x[1], y_min)
    is_top    = lambda x: np.isclose(x[1], y_max)

    solver.add_bc(0, is_bottom, zero)
    solver.add_bc(0, is_top,    zero)
    solver.add_bc(1, is_bottom, bot_disp)
    solver.add_bc(1, is_top,    top_disp,
                  measure_reaction=True, reaction_direction=(0.0, 1.0))

    solver.setup(
        check_stability=not args.no_stability,
        newton_options={
            "rel_tol": 1e-10, "abs_tol": 1e-8,
            "max_iter": 30, "div_rel_tol": 10.0,
            "switch_to_minres": True,
            "petsc_options": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        },
        stability_options={
            "nev": 5,
            "neg_tol": -1e-8,
            "petsc_options": {
                "st_ksp_type": "preonly",
                "st_pc_type": "lu",
            },
        }
    )

    # -----------------------------------------------------------------------
    # Load schedule: u/H(t) = t · max_strain;  total compression u = t · max_strain · H
    # -----------------------------------------------------------------------
    def load_schedule(t: float) -> None:
        u_total = t * args.max_strain * H
        bot_disp.value = +0.5 * u_total
        top_disp.value = -0.5 * u_total

    timestepper = TimeStepper(
        t_end=1.0,
        dt_init=1.0 / args.n_steps,
        dt_min=1e-8,
        dt_max=1.0 / args.n_steps,
        good_newton_steps=5,
    )

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    vtx = VTXManager(
        comm,
        os.path.join(args.output_dir, "solution.bp"),
        [solver.u_int, solver.F_func, solver.P_func, solver.J_func],
    )
    reaction_logger = ReactionForceLogger()

    solver.run(
        load_schedule,
        timestepper=timestepper,
        output_manager=vtx,
        reaction_logger=reaction_logger,
    )
    vtx.close()
    reaction_logger.save(
        comm,
        os.path.join(args.output_dir, "reaction.png"),
        os.path.join(args.output_dir, "reaction.csv"),
    )

    # -----------------------------------------------------------------------
    # Post-process: P_22 [kPa] vs u/H (rank 0 only).
    # disp = top_disp = -u/2  →  u/H = -2 disp / H.
    # Top reaction is negative under compression; P_22 = -R / W is positive.
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
            ax.set_title(f"DNS, {label} (H/W = {H / W:.2f})")
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
