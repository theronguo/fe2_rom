"""Macroscopic comparison: CANN surrogate vs Mooney–Rivlin reference.

Same boundary-value problem solved twice on the *same* mesh:

* **CANN** — the trained constitutive ANN as a closed-form ``dolfinx_materials``
  law (``CANNMaterial``) driven by the (material-pluggable) ``ch1.MacroSolver``;
* **Mooney–Rivlin** — the generated reference (eq. 18) as a UFL strain energy
  driven by the user's ``HyperelasticStabilitySolver``.

BVP: a 3D bar (``L×H×W``) clamped at ``x=0`` and pulled along x at ``x=L``
(lateral faces free) — uniaxial tension with end constraint, so the field is
heterogeneous near the clamp. Both materials share the same volumetric penalty
``κ`` so the comparison isolates the CANN's fidelity to the isochoric law.

Run inside the dolfinx-rve container:
    conda activate fe2_rom_env && python validation/cann/macro/run_comparison.py
"""
import fe2_rom  # noqa: F401  (dolfinx-before-torch import order)

import logging
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)   # validation/cann  (cann.py, validate_mooney_rivlin.py)
sys.path.insert(0, HERE)     # validation/cann/macro

from cann import CANN                                    # noqa: E402
import validate_mooney_rivlin as mrv                     # noqa: E402
from mooney_rivlin import GeneralizedMooneyRivlin        # noqa: E402
from cann_material import CANNMaterial                   # noqa: E402

from fe2_rom.hyperelastic_solver import (                # noqa: E402
    HyperelasticStabilitySolver, ReactionForceLogger, TimeStepper, VTXManager,
    broadcast_logger, setup_logging,
)
from fe2_rom.ch1.macrosolver import MacroSolver          # noqa: E402

comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
broadcast_logger("fe2_rom.ch1.macrosolver",
                 "fe2_rom.hyperelastic_solver.solver", level=logging.WARNING)
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# --- problem parameters ----------------------------------------------------
KAPPA = 5.0            # MPa, shared volumetric penalty
DELTA = 1.0            # max prescribed u_x at x=L  (nominal stretch 1+DELTA/L)
L, H, W = 4.0, 1.0, 1.0
NX, NY, NZ = 16, 4, 4
EPOCHS = int(os.environ.get("CANN_EPOCHS", "4000"))


def clamp_x0(x):
    return np.isclose(x[0], 0.0)


def face_xL(x):
    return np.isclose(x[0], L)


def owned_rel_l2(comm, a, b, V):
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    d = a[:n] - b[:n]
    num = comm.allreduce(float(d @ d), op=MPI.SUM)
    den = comm.allreduce(float(b[:n] @ b[:n]), op=MPI.SUM)
    return (num / den) ** 0.5


def main():
    # 1) train the CANN on Mooney–Rivlin data (sec. 4.1.1 protocol) -----------
    if comm.rank == 0:
        print("Training CANN on Mooney–Rivlin data ...")
    C, S = mrv.make_dataset(15)
    model = CANN(gdim=3, structure_tensors=None, hidden=(8, 8), incompressible=True)
    mrv.train(model, C, S, epochs=EPOCHS)
    model.eval()

    mesh = dmesh.create_box(
        comm, [np.array([0.0, 0.0, 0.0]), np.array([L, H, W])],
        [NX, NY, NZ], dmesh.CellType.tetrahedron)

    timestep = dict(t_end=1.0, dt_init=0.25, dt_min=1e-4, dt_max=0.25,
                    good_newton_steps=6)

    # 2) Mooney–Rivlin reference — the user's HyperelasticStabilitySolver ------
    if comm.rank == 0:
        print("\nSolving Mooney–Rivlin reference (HyperelasticStabilitySolver) ...")
    mr_mat = GeneralizedMooneyRivlin(kappa=KAPPA)
    mr = HyperelasticStabilitySolver(mesh, None, None, mr_mat, degree=1,
                                     enable_viz_fields=True)
    z = fem.Constant(mesh, PETSc.ScalarType(0.0))
    d_mr = fem.Constant(mesh, PETSc.ScalarType(0.0))
    for comp in (0, 1, 2):
        mr.add_bc(comp, clamp_x0, z)
    mr.add_bc(0, face_xL, d_mr, measure_reaction=True, reaction_direction=(1.0, 0.0, 0.0))
    mr.setup(check_stability=False)
    rl_mr = ReactionForceLogger()
    vtx_mr = VTXManager(comm, os.path.join(OUT, "mr.bp"),
                        [mr.u_int, mr.F_func, mr.P_func, mr.J_func])
    mr.run(lambda t: setattr(d_mr, "value", t * DELTA),
           timestepper=TimeStepper(**timestep),
           output_manager=vtx_mr, reaction_logger=rl_mr)
    vtx_mr.close()
    u_mr = mr.u.x.array.copy()

    # 3) CANN surrogate — material-pluggable ch1.MacroSolver ------------------
    if comm.rank == 0:
        print("\nSolving CANN surrogate (ch1.MacroSolver, pluggable material) ...")
    cann_mat = CANNMaterial(model, kappa=KAPPA, gdim=3)
    macro = MacroSolver(mesh, n_qp=2, gdim=3, degree=1, material=cann_mat,
                        check_stability=False)
    zc = fem.Constant(mesh, PETSc.ScalarType(0.0))
    d_cann = fem.Constant(mesh, PETSc.ScalarType(0.0))
    for comp in (0, 1, 2):
        macro.add_bc(comp, clamp_x0, zc)
    macro.add_bc(0, face_xL, d_cann, measure_reaction=True,
                 reaction_direction=(1.0, 0.0, 0.0))
    macro.setup()
    rl_cann = ReactionForceLogger()
    macro.solve(output_dir=os.path.join(OUT, "cann"),
                timestepper=TimeStepper(**timestep),
                loadhistory=lambda t: setattr(d_cann, "value", t * DELTA),
                output_variables=[macro.u], reaction_logger=rl_cann)
    u_cann = macro.u.x.array.copy()

    # 4) compare --------------------------------------------------------------
    rel_u = owned_rel_l2(comm, u_cann, u_mr, macro.V)
    if comm.rank == 0:
        print("\n================  CANN vs Mooney–Rivlin (macroscopic)  ================")
        print(f"  relative L2 displacement-field error : {rel_u*100:.3f} %")
        r_mr = np.array(rl_mr.forces)
        r_cann = np.array(rl_cann.forces)
        if r_mr.size and r_mr[-1] != 0:
            print(f"  final reaction  MR={r_mr[-1]:.5f}  CANN={r_cann[-1]:.5f}  "
                  f"(rel {abs(r_cann[-1]-r_mr[-1])/abs(r_mr[-1])*100:.3f} %)")
        _plot(rl_mr, rl_cann)


def _plot(rl_mr, rl_cann):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(matplotlib unavailable: {exc}; skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(rl_mr.displacements, rl_mr.forces, "-o", ms=4, lw=2,
            label="Mooney–Rivlin (HyperelasticStabilitySolver)")
    ax.plot(rl_cann.displacements, rl_cann.forces, "--s", ms=4, lw=1.6,
            label="CANN (MacroSolver)")
    ax.set_xlabel(r"prescribed displacement $u_x$ at $x=L$")
    ax.set_ylabel("reaction force $R_x$")
    ax.legend(fontsize=8)
    ax.set_title("Macroscopic CANN vs Mooney–Rivlin")
    fig.tight_layout()
    path = os.path.join(OUT, "macro_cann_vs_mr.png")
    fig.savefig(path, dpi=140)
    print(f"  saved {path}")


if __name__ == "__main__":
    main()
