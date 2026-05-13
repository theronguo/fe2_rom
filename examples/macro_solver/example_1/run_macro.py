"""FE² macro solver demo.

Mirrors fe2_rom/macro_solver/neo_hookean_demo.py but the constitutive law is
replaced by a periodic homogenization RVE solver at every macro quadrature
point.  The macro problem is assembled with dolfinx_materials' QuadratureMap /
NonlinearMaterialProblem; only the Material implementation changes.

Each macro qp owns its own RVE on MPI.COMM_SELF, so the demo can be run
serially or with mpirun (every macro rank handles its own qp population).

Run:
    conda activate fe2_rom_env
    python run_macro.py
"""
import os
# Avoid OpenBLAS/OpenMP oversubscription — many nested RVE solvers run serially.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
import ufl

from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector

from fe2_rom.hyperelastic_solver import (
    PeriodicHyperelasticHomogenizationSolver,
    NeoHookean,
    TimeStepper,
    VTXManager,
    setup_logging,
    broadcast_logger,
)
from fe2_rom.rve_rom.solver import RVESolver
from fe2_rom.macro_solver import RVEMaterial


comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.INFO)
logger = logging.getLogger("fe2_rom.macro_solver.example_1")
logger.addFilter(lambda r: comm.rank == 0)

VERBOSE_RVE = False
_RVE_LOGGERS = (
    "fe2_rom.hyperelastic_solver.solver",
    "fe2_rom.hyperelastic_solver.solvers",
    "fe2_rom.rve_rom.solver",
)
if VERBOSE_RVE:
    broadcast_logger(*_RVE_LOGGERS, level=logging.DEBUG)
else:
    for name in _RVE_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


# ===========================================================================
# RVE FACTORY
# ===========================================================================
# 3D periodic RVE from example_2 + its prebuilt ECM ROM.
RVE_MESH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "periodic_solver", "example_2", "mesh.msh",
)
RVE_ROM_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "periodic_solver", "example_2", "ecm",
)

# Microscopic Neo-Hookean parameters
E_micro,  nu_micro  = 3000.0, 0.30
mu_micro  = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))


def make_rve(rank, index):
    """Return a fresh ROM-based periodic RVE on COMM_SELF."""
    return RVESolver(
        mesh_path=RVE_MESH,
        rom_dir=RVE_ROM_DIR,
        material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
        comm=MPI.COMM_SELF,
        gdim=3,
        degree=1,
        output_dir=f"output/rve_{rank}_{index}",
        visualize_fields=["u_fluc"],
        average_fields=["P", "A"],
        newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
                        "max_iter": 50, "div_rel_tol": 10.0},
        timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-3,
                             "dt_max": 1.0, "good_newton_steps": 5},
        averages_only_final=True
    )


# def make_rve(rank, index):
#     """Full-order periodic RVE on COMM_SELF.

#     check_stability is disabled in the inner FE² loop: we only need the
#     response P, A at the given F, and eigenvector perturbations across macro
#     Newton iterations would make the returned tangent inconsistent.
#     """
#     return PeriodicHyperelasticHomogenizationSolver(
#         mesh_path=RVE_MESH,
#         comm=MPI.COMM_SELF,
#         gdim=3,
#         material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
#         degree=1,
#         output_dir=f"output/rve_{rank}_{index}",
#         visualize_fields=["u_fluc"],
#         check_stability=False,
#         average_fields=["P", "A"],
#         newton_options={"rel_tol": 1e-8, "abs_tol": 1e-6,
#                         "max_iter": 25, "switch_to_minres": False},
#         timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-3,
#                              "dt_max": 1.0, "good_newton_steps": 5},
#         save_snapshots=[],
#         averages_only_final=True,
#     )


# ===========================================================================
# MACRO MESH (kept small — FE² runs an RVE solve per qp per macro Newton iter)
# ===========================================================================
# 2×2×2 = 8 hex cells: parallelisable up to 8 MPI ranks (ParMETIS needs
# n_cells ≥ mpi_size to partition; a 1×1×1 mesh crashes under mpirun).
domain = mesh.create_unit_cube(comm, 4, 4, 4, cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.none)

V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
u  = fem.Function(V, name="displacement")
du = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)


# ===========================================================================
# BOUNDARY CONDITIONS — clamped left, prescribed uniaxial stretch on right
# ===========================================================================
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

x0_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], 0.0))
x1_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], 1.0))

bc_bot = fem.dirichletbc(
    np.zeros(3, dtype=np.float64),
    fem.locate_dofs_topological(V, fdim, x0_facets),
    V,
)

V_z, _ = V.sub(2).collapse()
disp_fn = fem.Function(V_z)
right_x_dofs = fem.locate_dofs_topological((V.sub(2), V_z), fdim, x1_facets)
bc_top = fem.dirichletbc(disp_fn, right_x_dofs, V.sub(2))

bcs = [bc_bot, bc_top]


# ===========================================================================
# RVE-BASED MATERIAL VIA QUADRATUREMAP
# ===========================================================================
material = RVEMaterial(make_rve)
qmap = QuadratureMap(domain, 2, material)   # 2×2×2 = 8 qp per hex

Id = ufl.Identity(3)
F_ufl  = nonsymmetric_tensor_to_vector(Id + ufl.grad(u))
dF_ufl = lambda w: ufl.derivative(F_ufl, u, w)
qmap.register_gradient("F", F_ufl)

P_vec = qmap.fluxes["PK1"]
Res = ufl.dot(P_vec, dF_ufl(v)) * qmap.dx
Jac = qmap.derivative(Res, u, du)


# ===========================================================================
# SNES (direct LU — the macro problem is tiny but its tangent is nonsymmetric)
# ===========================================================================
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_rtol": 1e-6,
    "snes_atol": 1e-8,
    "snes_max_it": 25,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

problem = NonlinearMaterialProblem(
    qmap, Res, u,
    bcs=bcs, J=Jac,
    petsc_options_prefix="fe2_macro_",
    petsc_options=petsc_options,
)


# ===========================================================================
# OUTPUT — BP4 time series (open in ParaView 5.11+ via the ADIOS2 reader)
# ===========================================================================
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
vtx = VTXManager(comm, os.path.join(output_dir, "macro.bp"), [u])


# ===========================================================================
# ADAPTIVE LOAD STEPPING (matches the other solvers' timestepper pattern).
#
# Pseudo-time t ∈ [0, 1]; the prescribed displacement is interpolated
# linearly between 0 and disp_total.  SNES failure → halve dt and retry;
# success → accept and grow dt back toward dt_max.  On revert we restore
# the macro displacement to the last accepted state — RVE state warm-starts
# from wherever it currently sits, which is fine for path-independent
# constitutive laws since each RVE call solves to equilibrium at the target F.
# ===========================================================================
disp_total = -0.25   # 25 % uniaxial compression
timestepper = TimeStepper(
    t_end=1.0, dt_init=0.2, dt_min=1e-3, dt_max=0.2, good_newton_steps=5,
)

u_last = fem.Function(V)   # last accepted macro displacement, for revert

vtx.write(0.0)
try:
    while not timestepper.finished:
        trial_t = timestepper.step_forward()
        logger.info("── Step  t=%.5f  dt=%.2e", trial_t, timestepper.dt)

        disp_fn.x.array[:] = disp_total * trial_t

        material.step_failed = False
        material.failure_reason = ""
        try:
            problem.solve()
            reason  = problem.solver.getConvergedReason()
            n_iters = problem.solver.getIterationNumber()
        except PETSc.Error as exc:
            reason, n_iters = -1, 0

        if material.step_failed:
            logger.warning("%s", material.failure_reason)
            reason = -1

        if reason > 0:
            timestepper.accept(n_iters)
            u_last.x.array[:] = u.x.array
            u_last.x.scatter_forward()
            material.commit()
            logger.info(
                "   SNES converged in %d iteration(s)  disp=%+.6f",
                n_iters, disp_total * trial_t,
            )
            vtx.write(trial_t)
        else:
            ok = timestepper.reject()
            u.x.array[:] = u_last.x.array
            u.x.scatter_forward()
            if not ok:
                logger.error(
                    "Minimum time step dt=%.2e reached — stopping.",
                    timestepper.dt_min,
                )
                break
            logger.warning(
                "SNES did not converge in %d iteration(s) (reason=%d) — halving dt to %.2e",
                n_iters, reason, timestepper.dt,
            )
finally:
    vtx.close()
