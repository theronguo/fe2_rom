"""Full-order first-order (CH1) periodic hyperelastic homogenization solver."""
import os
import logging
from typing import Callable

import numpy as np
from dolfinx import fem, io, mesh as dmesh
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import dolfinx_mpc

from fe2_rom.ch1.averages import (
    EffectiveAbar,
    EffectiveAbarReduced,
    EffectiveFbar,
    EffectivePbar,
    HomogenizationContext,
    resolve_average_quantities,
)
from fe2_rom.ch1.objectivity import (
    objective_transform_pbar,
    polar_derivatives,
    symmetric_basis_tensors,
)
from fe2_rom.ch1.constraints import LinearConstraint, ZeroVolumeAverage
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.solvers import NewtonSolverFE2
from fe2_rom.hyperelastic_solver.forms import basis_tensor_ufl, build_homogenization_weak_form
from fe2_rom.hyperelastic_solver.logging_utils import silence_c_stdout
from fe2_rom.hyperelastic_solver.material import MaterialModel
from fe2_rom.hyperelastic_solver.output import VTXManager
from fe2_rom.hyperelastic_solver.stability import (
    StabilityAnalyzer,
    apply_eigenmode_perturbation,
    mesh_characteristic_length,
    solve_smallest_eigenpairs,
)
from fe2_rom.hyperelastic_solver.timestepping import TimeStepper

logger = logging.getLogger(__name__)


class MicroSolver:
    """Periodic hyperelastic homogenization with pluggable effective quantities.

    Periodic ties are enforced via ``dolfinx_mpc`` for faces and corners
    (all non-max corners are slaved to the max corner). The Newton step is
    the standard ``NewtonSolverFE2`` path with CG/MINRES + GAMG
    (no saddle-point augmentation).

    Effective quantities and tangent moduli are computed through the modular
    ``AverageQuantity`` interface; ``__call__`` returns
    ``list[dict[str, ...]]`` — one dict per accepted load step, keyed by
    each quantity's ``name``.

    Subclass hooks (``_setup_phi``, ``_setup_macro_vars``,
    ``_build_u_total_extra``, ``_build_macro_var_rhs_forms``,
    ``_make_default_average_quantities``, etc.) allow derived classes to add
    additional macro variables, ansatz contributions, constraints, and
    effective quantities — used by the micromorphic subclass.
    """

    def __init__(self, mesh_path, comm, gdim,
                 material: MaterialModel, *,
                 degree: int = 1,
                 quadrature_degree: int | None = None,
                 output_dir: str = "output",
                 check_stability: bool = True,
                 perturb_post_buckling: bool = True,
                 pert_amplitude_init: float = 1e-2,
                 visualize_fields: list[str] | None = None,
                 average_quantities: list | None = None,
                 stability_options: dict | None = None,
                 newton_options: dict | None = None,
                 timestepper_options: dict | None = None,
                 save_snapshots: list[str] | None = None,
                 averages_only_final: bool = False,
                 corner_periodic: bool = False,
                 constraints: "list[LinearConstraint] | None" = None,
                 rve_volume: float | None = None,
                 lattice_vectors: "np.ndarray | None" = None,
                 objective_reduction: bool = False,
                 ) -> None:

        newton_options = newton_options if newton_options is not None else {
            "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 10, "max_iter_instab": 30,
            "switch_to_minres": False, "div_rel_tol": 10.0,
        }
        timestepper_options = timestepper_options if timestepper_options is not None else {
            "t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5, "dt_max": 1.0, "good_newton_steps": 7,
        }
        stability_options = stability_options if stability_options is not None else {
            "nev": 5, "neg_tol": -1e-12,
        }
        if visualize_fields is None:
            visualize_fields = ["u_fluc"]
        if save_snapshots is None:
            save_snapshots = []

        # ---- Mesh ----
        self.comm = comm
        with silence_c_stdout():
            mesh_data = io.gmsh.read_from_msh(mesh_path, self.comm, 0, gdim=gdim)
        self._mesh = mesh_data.mesh
        self._cell_tags = mesh_data.cell_tags
        self._facet_tags = mesh_data.facet_tags
        self._mesh.topology.create_connectivity(self._mesh.topology.dim - 1, self._mesh.topology.dim)
        # Quadrature degree for every form built off ``self.dx`` (residual,
        # Jacobian, tangent-RHS adjoints, volume, effective averages, the
        # fluctuation constraints and the H¹ stability form). ``None`` keeps
        # DOLFINx's automatic estimate, which over-integrates higher-order
        # elements (e.g. 6 points for a tri6, 14+ for a tet10); pass an explicit
        # degree to use the cheaper rule that is accurate enough in practice.
        self._quadrature_degree = quadrature_degree
        dx_metadata = (
            {"quadrature_degree": int(quadrature_degree)}
            if quadrature_degree is not None else None
        )
        self.dx = ufl.Measure("dx", domain=self._mesh,
                              subdomain_data=self._cell_tags, metadata=dx_metadata)
        self.gdim = gdim

        # Objectivity (F̄ = R U) reduction. When enabled, ``__call__`` drives the
        # RVE with the symmetric stretch ``U`` and reconstructs the lab-frame
        # ``P̄ = R P̃`` / ``dP̄/dF̄`` analytically, using only the 6 (3D) / 3 (2D)
        # symmetric adjoint directions instead of ``gdim²``.
        self._objective = bool(objective_reduction)

        self.mins, self.maxs = self._compute_domain_bounds()
        self.length_scale = (self.maxs - self.mins).max()
        logger.debug("Mesh loaded: %d cells, %d facets, gdim=%d",
                     self._mesh.topology.index_map(self._mesh.topology.dim).size_global,
                     self._mesh.topology.index_map(self._mesh.topology.dim - 1).size_global,
                     gdim)
        logger.debug("Domain bounds: x [%.3f, %.3f]", self.mins[0], self.maxs[0])
        logger.debug("Domain bounds: y [%.3f, %.3f]", self.mins[1], self.maxs[1])
        if gdim == 3:
            logger.debug("Domain bounds: z [%.3f, %.3f]", self.mins[2], self.maxs[2])

        self._averages_only_final = averages_only_final
        self._material = material
        # When False, an instability is not perturbed onto the buckled branch;
        # instead the step is rejected and dt halved, so the solve approaches the
        # bifurcation as closely as possible (used by the φ-extraction "lba"
        # strategy, which then does a linear buckling analysis there).
        self._perturb_post_buckling = perturb_post_buckling
        # Default initial eigenmode-kick amplitude for post-buckling traversal,
        # used by ``__call__`` when not overridden per-call. Exposed so the FE²
        # driver / RVE factory can tune it (the solver default 1e-2 overshoots
        # Newton's basin for thin-strut RVEs; ~1e-3 traverses isolated modes).
        self._pert_amplitude_init = pert_amplitude_init
        # F̄ at the most recent accepted (stable) step — the near-critical state
        # the "lba" strategy hands to compute_buckling_spectrum. Re-initialised at
        # the start of every __call__ (F_bar is built later in __init__).
        self._last_converged_Fbar = np.eye(self.gdim, dtype=PETSc.ScalarType)

        # ---- Function space and state fields ----
        self._degree = degree
        self.V = fem.functionspace(self._mesh, ("Lagrange", degree, (self.gdim,)))
        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs

        self.u = fem.Function(self.V)
        self._u_last = fem.Function(self.V)
        self._u_conv = fem.Function(self.V)
        self._du = fem.Function(self.V)
        self._eigenfunction = fem.Function(self.V)
        self.F_bar = fem.Constant(self._mesh, np.eye(self.gdim, dtype=PETSc.ScalarType))
        self.F_bar_conv = np.eye(self.gdim, dtype=PETSc.ScalarType)
        logger.debug("Functions set up with %d global DOFs", n_dofs)

        # ---- Periodic BCs + MPC ----
        self._corner_periodic = corner_periodic
        self._lattice_vectors = None
        self._polygon_periodic = False
        if lattice_vectors is not None:
            lv = np.asarray(lattice_vectors, dtype=float)
            if self.gdim != 2:
                raise ValueError("lattice_vectors is only supported for gdim==2.")
            if lv.shape != (2, 2):
                raise ValueError(
                    f"lattice_vectors must have shape (2, 2) (rows a1, a2), got {lv.shape}."
                )
            if self._corner_periodic:
                raise ValueError(
                    "lattice_vectors and corner_periodic are mutually exclusive; the "
                    "polygon path already uses the full-periodicity + ZeroVolumeAverage "
                    "gauge-fix regime."
                )
            det = float(np.linalg.det(lv))
            if abs(det) < 1e-14 * (float(np.linalg.norm(lv)) ** 2 + 1e-30):
                raise ValueError("lattice_vectors are (near-)linearly dependent.")
            self._lattice_vectors = lv
            self._polygon_periodic = True
        bcs, self.mpc = self._setup_periodic_bcs_and_mpc()
        self._bcs = bcs
        logger.debug("Periodic BCs set up with %d slave points",
                     self.comm.allreduce(len(self.mpc.slaves), op=MPI.SUM))

        # ---- Subclass hooks: φᵢ and the macro-variable dict ----
        self._setup_phi()
        self._setup_macro_vars()

        # Characteristic mesh extent — fallback length scale for the
        # eigenmode perturbation when |u| is ~0 (e.g. first load step).
        self._char_length = mesh_characteristic_length(self._mesh)

        # ---- Stability analyzer (optional) ----
        if check_stability:
            n_skip_default = self._count_zero_modes()
            n_skip = stability_options.pop("n_skip_eigenvalues", n_skip_default)
            self._stability = StabilityAnalyzer(
                self.comm, n_skip_eigenvalues=n_skip, **stability_options,
            )
            logger.debug("Stability checks enabled (skipping %d gauge eigenvalues); options: %s",
                         n_skip, stability_options)
            if "switch_to_minres" in newton_options:
                if newton_options["switch_to_minres"] is False:
                    logger.debug("Overriding newton_options['switch_to_minres'] to True for stability checks.")
            newton_options["switch_to_minres"] = True
        else:
            self._stability = None

        # ---- Weak form (with optional u_total_extra from subclass) ----
        u_total_extra = self._build_u_total_extra()
        (R_form, J_form, F_var, P_ufl, J_ufl, W_ufl, A_ufl, u_total,
         build_tangent_rhs_forms) = build_homogenization_weak_form(
            self._mesh, self.V, self.u, self.F_bar, self._material,
            u_total_extra=u_total_extra, dx=self.dx,
        )

        # ---- Adjoint-RHS forms per macro variable (subclass extends) ----
        self._macro_var_rhs_forms = self._build_macro_var_rhs_forms(build_tangent_rhs_forms)

        # ---- Constraint forms (projected Newton solve) ----
        constraint_forms = self._build_constraint_forms(constraints)

        # ---- Newton solver (NewtonSolverFE2 with periodic MPC ties) ----
        self._newton = NewtonSolverFE2(
            self.comm, R_form, J_form, self.u, self._du, self._bcs, self.mpc,
            constraint_forms=constraint_forms,
            **newton_options,
        )
        logger.debug("Newton solver initialized with options: %s", newton_options)

        # ---- Time stepper ----
        self._timestepper = TimeStepper(**timestepper_options)
        logger.debug("Time stepper initialized with options: %s", timestepper_options)

        # ---- Visualization ----
        self._setup_visualization(visualize_fields, F_var, P_ufl, J_ufl, W_ufl, u_total, output_dir)

        # ---- Volume + homogenization context ----
        # Macroscopic averaging is (1/|Q|) ∫_Ω · dX with |Q| the periodic-cell
        # volume — voids counted.  For arbitrary (e.g. hexagonal) cells the
        # caller MUST pass ``rve_volume``; the bounding box would be wrong.
        # If not provided we fall back to ∫_Ω 1 dx = |Ω_solid|, which equals
        # |Q| only for non-porous RVEs.
        if rve_volume is None:
            vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
            self._vol_global = float(self.comm.allreduce(vol_local, op=MPI.SUM))
        else:
            self._vol_global = float(rve_volume)
        self._context = HomogenizationContext(
            mesh=self._mesh, V=self.V, dx=self.dx, comm=self.comm,
            vol_global=self._vol_global,
            F_var=F_var, P_ufl=P_ufl, A_ufl=A_ufl, W_ufl=W_ufl,
            u=self.u, u_total=u_total,
            macro_vars=self.macro_vars,
            phi=self._phi,
        )

        # ---- Average quantities ----
        if average_quantities is None:
            average_quantities = self._make_default_average_quantities()
        self._average_quantities = resolve_average_quantities(
            average_quantities, self._string_key_factories())
        if self._objective:
            self._average_quantities = self._objectivize_quantities(
                self._average_quantities)
        for q in self._average_quantities:
            q.setup(self._context)

        # ---- Snapshot saving ----
        self.output_dir = output_dir
        self.save_snapshots = save_snapshots
        # Cache for the most recent macro-variable sensitivities ∂w/∂μ_k,
        # populated by ``_collect_averages``. Keyed by macro-var name
        # (e.g. "Fbar", "v", "g") -> list[fem.Function]. Consumed by the
        # ``dw_d{name}`` snapshot path.
        self._last_sensitivities: dict | None = None

        # ---- Optional A (stiffness) snapshot helper ----
        # A is a rank-4 tensor field (∂P/∂F). We only build the DG storage
        # when actually requested, since it costs gdim^4 components per cell.
        self.A_func = None
        self._A_expr = None
        if "A" in save_snapshots:
            AA = fem.functionspace(
                self._mesh, ("DG", 1, (self.gdim, self.gdim, self.gdim, self.gdim)),
            )
            self.A_func = fem.Function(AA, name="A")
            self._A_expr = fem.Expression(A_ufl, AA.element.interpolation_points)

        logger.debug("Snapshot fields: %s", save_snapshots)
        logger.debug("Setup complete (n_dofs=%d)", n_dofs)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _setup_phi(self) -> None:
        self._phi: list[fem.Function] = []

    def _setup_macro_vars(self) -> None:
        self.macro_vars: dict = {"Fbar": self.F_bar}

    def _build_u_total_extra(self):
        return None

    def _build_macro_var_rhs_forms(self, build_tangent_rhs_forms) -> dict:
        return {"Fbar": build_tangent_rhs_forms(self._fbar_adjoint_directions())}

    def _fbar_adjoint_directions(self) -> list:
        """UFL 2-tensor directions ``∂F/∂F̄_k`` for the ``Fbar`` adjoint set.

        Objective reduction: the 6 (3D) / 3 (2D) *symmetric* basis tensors
        (sensitivities w.r.t. the stretch ``U``). Otherwise: the full ``gdim²``
        single-entry basis tensors.
        """
        gdim = self.gdim
        if self._objective:
            return [ufl.as_matrix(S.tolist())
                    for S in symmetric_basis_tensors(gdim)]
        return [basis_tensor_ufl(gdim, i, j)
                for i in range(gdim) for j in range(gdim)]

    def _objectivize_quantities(self, qs: list) -> list:
        """Swap effective quantities for their U-frame reduced variants when the
        objectivity reduction is active. Subclasses extend this for their own
        ``/dF̄`` tangent blocks."""
        out = []
        for q in qs:
            if type(q) is EffectiveAbar:
                out.append(EffectiveAbarReduced())
            else:
                out.append(q)
        return out

    def _make_default_average_quantities(self) -> list:
        return [EffectiveFbar(), EffectivePbar(), EffectiveAbar()]

    def _string_key_factories(self) -> dict:
        """Solver-specific ``{key: zero-arg callable}`` string-key constructors,
        extending ``STRING_KEY_MAP`` for quantities that need solver-held state
        (the micromorphic subclasses register their φ-bound ``"Pi"``,
        ``"Lambda"`` and tangent-block keys here)."""
        return {}

    def _build_constraint_forms(self, constraints: "list[LinearConstraint] | None") -> list:
        if constraints is None:
            constraints = (
                [ZeroVolumeAverage()]
                if (self._corner_periodic or self._polygon_periodic)
                else []
            )
        if not constraints:
            return []
        forms = []
        for c in constraints:
            c_forms, _ = c.build(self.V, self.dx, self._mesh, self.mpc)
            forms.extend(c_forms)
        logger.debug("Constraint forms: %d rows from %d constraint object(s)",
                     len(forms), len(constraints))
        return forms

    def _restore_trial_state(self) -> None:
        return

    def _update_macro_load_schedule(self, t: float) -> None:
        return

    def _commit_extra_state(self) -> None:
        return

    def _count_zero_modes(self) -> int:
        return self.gdim + 1 if (self._corner_periodic or self._polygon_periodic) else 0

    # ------------------------------------------------------------------
    # Mesh / MPC helpers
    # ------------------------------------------------------------------

    def _compute_domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        coords = self._mesh.geometry.x
        mins_local = np.min(coords[:, :self.gdim], axis=0)
        maxs_local = np.max(coords[:, :self.gdim], axis=0)
        mins = np.array(
            [self.comm.allreduce(float(mins_local[i]), op=MPI.MIN) for i in range(self.gdim)],
            dtype=float,
        )
        maxs = np.array(
            [self.comm.allreduce(float(maxs_local[i]), op=MPI.MAX) for i in range(self.gdim)],
            dtype=float,
        )
        return mins, maxs

    @staticmethod
    def _make_axis_map(axis: int, target_value: float) -> Callable:
        def axis_map(x):
            y = x.copy()
            y[axis] = target_value
            return y
        return axis_map

    @staticmethod
    def _make_periodic_slave_selector(axis: int, mins: np.ndarray, maxs: np.ndarray,
                                      tol: float, exclude_axes: tuple[int, ...],
                                      corners: list[np.ndarray]) -> Callable:
        def selector(x):
            mask = np.isclose(x[axis], mins[axis], atol=tol, rtol=0.0)
            for ex_axis in exclude_axes:
                mask &= (x[ex_axis] > mins[ex_axis] + tol)
                mask &= (x[ex_axis] < maxs[ex_axis] - tol)
            for corner in corners:
                on_corner = np.ones(x.shape[1], dtype=bool)
                for ax in range(len(corner)):
                    on_corner &= np.isclose(x[ax], corner[ax], atol=tol, rtol=0.0)
                mask &= ~on_corner
            return mask
        return selector

    @staticmethod
    def _make_corner_selector(corner_coords: np.ndarray, tol: float) -> Callable:
        def selector(x):
            mask = np.ones(x.shape[1], dtype=bool)
            for axis, value in enumerate(corner_coords):
                mask &= np.isclose(x[axis], value, atol=tol, rtol=0.0)
            return mask
        return selector

    @staticmethod
    def _make_corner_map(master_coords: np.ndarray) -> Callable:
        def corner_map(x):
            y = x.copy()
            for axis, value in enumerate(master_coords):
                y[axis] = value
            return y
        return corner_map

    @staticmethod
    def _corner_coordinates(mins: np.ndarray, maxs: np.ndarray) -> list[np.ndarray]:
        dim = len(mins)
        corners: list[np.ndarray] = []
        for bits in range(1 << dim):
            coord = np.empty(dim, dtype=float)
            for axis in range(dim):
                coord[axis] = maxs[axis] if (bits >> axis) & 1 else mins[axis]
            corners.append(coord)
        return corners

    @staticmethod
    def _locate_corner_dofs(V, mins: np.ndarray, maxs: np.ndarray, tol: float) -> np.ndarray:
        dim = len(mins)

        def corner_selector(x):
            mask = np.ones(x.shape[1], dtype=bool)
            for axis in range(dim):
                on_min = np.isclose(x[axis], mins[axis], atol=tol, rtol=0.0)
                on_max = np.isclose(x[axis], maxs[axis], atol=tol, rtol=0.0)
                mask &= on_min | on_max
            return mask

        return fem.locate_dofs_geometrical(V, corner_selector)

    def _setup_periodic_bcs_and_mpc(self) -> tuple[list, dolfinx_mpc.MultiPointConstraint]:
        if self.gdim not in (2, 3):
            raise ValueError(
                f"Periodic homogenization supports only 2D rectangle or 3D cuboid, got dim={self.gdim}."
            )
        tol = 1e-8 * max(1.0, float(np.max(self.maxs - self.mins)))

        if self._polygon_periodic:
            return self._setup_polygon_periodic_mpc(tol)

        if self._corner_periodic:
            bcs: list = []
        else:
            corner_dofs = self._locate_corner_dofs(self.V, self.mins, self.maxs, tol)
            u_zero = fem.Constant(self._mesh, np.zeros(self.gdim, dtype=PETSc.ScalarType))
            bcs = [fem.dirichletbc(u_zero, corner_dofs, self.V)]

        mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        for axis in range(self.gdim):
            selector = self._make_periodic_slave_selector(
                axis=axis, mins=self.mins, maxs=self.maxs, tol=tol,
                exclude_axes=tuple(range(axis)), corners=self._corner_coordinates(self.mins, self.maxs)
            )
            axis_map = self._make_axis_map(axis, self.maxs[axis])
            mpc.create_periodic_constraint_geometrical(self.V, selector, axis_map, bcs)

        if self._corner_periodic:
            master_corner = self.maxs.copy()
            corner_map = self._make_corner_map(master_corner)
            for corner_coords in self._corner_coordinates(self.mins, self.maxs):
                if np.allclose(corner_coords, master_corner, atol=tol, rtol=0.0):
                    continue
                selector = self._make_corner_selector(corner_coords, tol)
                mpc.create_periodic_constraint_geometrical(self.V, selector, corner_map, bcs)

        mpc.finalize()

        return bcs, mpc

    # ------------------------------------------------------------------
    # Arbitrary 2D polygon periodicity (geometric, lattice-vector driven)
    # ------------------------------------------------------------------

    def _setup_polygon_periodic_mpc(self, tol: float) -> tuple[list, dolfinx_mpc.MultiPointConstraint]:
        """Periodic MPC for an arbitrary 2D periodic polygon (e.g. a hexagon).

        Opposite boundary edges/vertices are auto-paired geometrically from the
        two lattice vectors ``a1, a2`` (``self._lattice_vectors``) and the actual
        mesh boundary. Slaves on the ``-t`` side of each edge pair are tied to the
        matching master on the ``+t`` side; polygon vertices are tied per
        lattice-equivalence orbit (depth-1 star to one representative). No
        Dirichlet BCs are applied — the ``gdim`` rigid-translation gauge is removed
        downstream by the default ``ZeroVolumeAverage`` constraint.

        Prerequisites (failure to meet these silently corrupts the homogenized
        response):

        * The mesh must be generated with ``gmsh.model.mesh.setPeriodic`` between
          each opposite edge pair (matching ``a1, a2``), so a slave node has an
          exact translated master node — otherwise the geometric tie finds no
          master (under-constrained) or an interpolated one (wrong condition).
        * Holes must be strictly interior; their lattice translates then miss the
          boundary and the hole-boundary dofs are auto-excluded.
        * ``rve_volume`` must be passed for porous cells (the exact cell area
          ``|Q|``); the ``∫ 1 dx`` fallback is only correct for non-porous cells.
        """
        bcs: list = []
        a1, a2 = self._lattice_vectors[0], self._lattice_vectors[1]
        positives, shell = self._lattice_translations(a1, a2, tol)

        B = self._gather_boundary_dof_coords()
        if B.shape[0] == 0:
            raise RuntimeError("No boundary dofs found for polygon periodic BCs.")
        contains = self._make_point_membership(B, tol)

        edge_pts, vertex_pts, n_untied = self._classify_boundary_points(B, shell, contains)
        if edge_pts.shape[0] + vertex_pts.shape[0] == 0:
            raise RuntimeError(
                "lattice_vectors do not match the mesh boundary: no boundary point "
                "has a lattice-translated partner on the boundary. Check that "
                "lattice_vectors are the true primitive translations of the mesh."
            )
        logger.debug(
            "Polygon periodicity: %d boundary nodes (%d edge, %d vertex, %d untied/holes); "
            "%d edge-pair translations",
            B.shape[0], edge_pts.shape[0], vertex_pts.shape[0], n_untied, len(positives),
        )

        mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        gdim = self.gdim

        # Edge-interior ties: for each positive translation, slaves are the
        # edge-interior points whose +t translate is also on the boundary (the
        # opposite-edge master). Masters live on the +t side and are never slaves.
        for t_k in positives:
            on_master_side = contains((edge_pts + t_k[:gdim]).T)
            slave_pts = edge_pts[on_master_side]
            if slave_pts.shape[0] == 0:
                continue
            indicator = self._make_point_membership(slave_pts, tol)
            relation = self._make_translation_relation(t_k, gdim)
            mpc.create_periodic_constraint_geometrical(self.V, indicator, relation, bcs)

        # Vertex ties: per lattice-equivalence orbit, depth-1 star to one rep.
        self._build_vertex_orbit_ties(mpc, vertex_pts, bcs, tol)

        mpc.finalize()
        return bcs, mpc

    def _gather_boundary_dof_coords(self) -> np.ndarray:
        """Allgathered global array of owned boundary dof (block) coordinates,
        shape ``(M, gdim)``. Restricts to owned blocks before gathering so a node
        shared across ranks is not double-counted."""
        tdim = self._mesh.topology.dim
        self._mesh.topology.create_connectivity(tdim - 1, tdim)
        ext_facets = dmesh.exterior_facet_indices(self._mesh.topology)
        bdofs = fem.locate_dofs_topological(self.V, tdim - 1, ext_facets)
        coords = self.V.tabulate_dof_coordinates()
        n_owned = self.V.dofmap.index_map.size_local
        owned = bdofs[bdofs < n_owned]
        local_pts = coords[owned, :self.gdim]
        gathered = self.comm.allgather(local_pts)
        parts = [g for g in gathered if len(g)]
        return np.vstack(parts) if parts else np.empty((0, self.gdim), dtype=float)

    @staticmethod
    def _lattice_translations(a1: np.ndarray, a2: np.ndarray, tol: float
                              ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return ``(positives, shell)``: the near-neighbor lattice translations
        ``{n1*a1 + n2*a2 : n1,n2 in {-1,0,1}} \\ {0}`` (8 vectors, ``shell``), and
        the 4 "positive" representatives (one per ±pair, ``positives``) in a fixed
        deterministic order. Suffices for Wigner-Seitz-type cells where opposite
        faces differ by a single near-neighbor lattice vector."""
        a1 = np.asarray(a1, dtype=float)
        a2 = np.asarray(a2, dtype=float)
        shell = [n1 * a1 + n2 * a2
                 for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)
                 if not (n1 == 0 and n2 == 0)]
        positives = [t for t in shell
                     if t[0] > tol or (abs(t[0]) <= tol and t[1] > tol)]
        positives.sort(key=lambda t: (float(np.arctan2(t[1], t[0])), float(t @ t)))
        return positives, shell

    @staticmethod
    def _make_point_membership(points: np.ndarray, tol: float) -> Callable:
        """Return ``contains(P)`` mapping a coordinate array ``P`` of shape
        ``(>=gdim, n)`` to a boolean mask ``(n,)`` that is True where a column of
        ``P`` coincides (within ``tol``) with a row of ``points`` (shape
        ``(M, gdim)``)."""
        gdim = points.shape[1]
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)

            def contains(P):
                d, _ = tree.query(np.ascontiguousarray(P[:gdim].T),
                                  k=1, distance_upper_bound=tol)
                return np.isfinite(d)
            return contains
        except Exception:  # pragma: no cover - scipy expected in the env
            scale = max(1.0, float(np.max(np.abs(points))) if points.size else 1.0)
            decimals = max(0, int(round(-np.log10(tol / scale))) - 1)
            keyset = set(map(tuple, np.round(points / scale, decimals)))

            def contains(P):
                keys = np.round(P[:gdim].T / scale, decimals)
                return np.fromiter((tuple(r) in keyset for r in keys),
                                   dtype=bool, count=keys.shape[0])
            return contains

    @staticmethod
    def _make_translation_relation(t: np.ndarray, gdim: int) -> Callable:
        """Map a slave coordinate array to ``x + t`` (the master side)."""
        t = np.asarray(t, dtype=float)

        def relation(x):
            y = x.copy()
            for d in range(gdim):
                y[d] = x[d] + t[d]
            return y
        return relation

    def _classify_boundary_points(self, B: np.ndarray, shell: list[np.ndarray],
                                  contains: Callable) -> tuple[np.ndarray, np.ndarray, int]:
        """Split boundary nodes ``B`` into edge-interior vs. polygon vertices by
        counting how many near-neighbor translates land back on the boundary:
        exactly 1 -> edge-interior, >=2 -> vertex, 0 -> untied (hole boundary)."""
        counts = np.zeros(B.shape[0], dtype=int)
        for t in shell:
            counts += contains((B + t[:self.gdim]).T).astype(int)
        edge_pts = B[counts == 1]
        vertex_pts = B[counts >= 2]
        n_untied = int(np.count_nonzero(counts == 0))
        return edge_pts, vertex_pts, n_untied

    def _build_vertex_orbit_ties(self, mpc: dolfinx_mpc.MultiPointConstraint,
                                 vertex_pts: np.ndarray, bcs: list, tol: float) -> None:
        """Tie polygon vertices per lattice-equivalence orbit. Two vertices share
        an orbit iff their difference is an integer combination of ``a1, a2``. Each
        orbit ties its members to one representative (depth-1 star); the rep is a
        genuine non-slave dof. Distinct orbits stay independent (e.g. a hexagon's
        two honeycomb sublattices), coupled only through the bulk stiffness."""
        n = vertex_pts.shape[0]
        if n == 0:
            return
        A = np.column_stack([self._lattice_vectors[0], self._lattice_vectors[1]])  # (2, 2)

        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                d = vertex_pts[j, :self.gdim] - vertex_pts[i, :self.gdim]
                c, *_ = np.linalg.lstsq(A, d, rcond=None)
                if (np.allclose(A @ c, d, atol=tol, rtol=0.0)
                        and np.allclose(c, np.round(c), atol=1e-6, rtol=0.0)):
                    parent[find(i)] = find(j)

        orbits: dict[int, list[int]] = {}
        for i in range(n):
            orbits.setdefault(find(i), []).append(i)

        logger.debug("Polygon vertex orbits: %d orbit(s) over %d vertices",
                     len(orbits), n)

        for members in orbits.values():
            rep_idx = max(members, key=lambda k: tuple(vertex_pts[k, :self.gdim]))
            rep = vertex_pts[rep_idx, :self.gdim]
            for k in members:
                if k == rep_idx:
                    continue
                selector = self._make_corner_selector(vertex_pts[k, :self.gdim], tol)
                rep_map = self._make_corner_map(rep)
                mpc.create_periodic_constraint_geometrical(self.V, selector, rep_map, bcs)

    # ------------------------------------------------------------------
    # Linear buckling spectrum (eigenvalues of the tangent K)
    # ------------------------------------------------------------------

    def compute_buckling_spectrum(self, n_eig: int, *,
                                  Fbar: "np.ndarray | None" = None,
                                  tol: float = 1e-6,
                                  slepc_options: "dict | None" = None,
                                  n_skip: "int | None" = None,
                                  visualize_modes: bool = False,
                                  modes_filename: str = "buckling_modes.bp",
                                  save_modes: bool = False,
                                  save_eigvals: bool = False,
                                  return_modes: bool = False):
        """Linear buckling spectrum of the tangent ``K`` at a reference state.

        Assembles ``K`` at the reference macro state (``Fbar=None`` ⇒ the current
        state, undeformed ``F̄=I`` by default; otherwise the RVE is first solved
        to ``Fbar``), solves the smallest ``n_eig + n_skip`` eigenpairs of
        ``K φ = λ φ`` (SLEPc shift-invert at ``σ=0``), backsubstitutes the periodic
        ties and scales each eigenvector to unit H¹ norm. Logs the ``n_skip``
        skipped null/gauge eigenvalues and the ``n_eig`` physical ones.

        Optional outputs:

        * ``visualize_modes`` → ``output_dir/modes_filename`` (one ParaView
          timestep per mode);
        * ``save_modes`` → ``output_dir/snapshots/phi_<i>.npy`` mode arrays,
          loadable into a micromorphic basis (see
          ``mm.MicroSolver.load_buckling_modes``);
        * ``save_eigvals`` (or ``save_modes``) →
          ``output_dir/snapshots/buckling_eigvals.npy``.

        Returns the physical eigenvalues (``NaN`` where SLEPc did not converge),
        or ``(eigvals, modes)`` with the in-memory eigenvector ``fem.Function``s
        when ``return_modes=True``. The micromorphic ``compute_linear_buckling_modes``
        uses ``return_modes=True`` to populate its φ basis without a disk
        round-trip; with ``N=0`` (no basis) this is a pure spectrum probe.

        ``n_skip`` defaults to ``self._count_zero_modes()`` (``gdim+1`` under full
        periodicity: rigid-body translations + one MPC gauge mode).
        """
        if n_eig <= 0:
            raise ValueError(f"n_eig must be positive, got {n_eig}")
        if n_skip is None:
            n_skip = self._count_zero_modes()

        if Fbar is not None:
            logger.info("Buckling spectrum: reference solve at F̄ =\n%s", Fbar)
            self(Fbar)

        K = self._newton.assemble_stiffness()
        try:
            nev_total = n_eig + n_skip
            eps, n_conv = solve_smallest_eigenpairs(
                K, self.comm, nev=nev_total, tol=tol, petsc_options=slepc_options,
            )
            if n_conv < nev_total:
                logger.warning(
                    "Buckling spectrum: requested %d eigenpairs (%d physical + %d "
                    "skipped), SLEPc converged %d", nev_total, n_eig, n_skip, n_conv,
                )

            skipped = [eps.getEigenvalue(i).real for i in range(min(n_skip, n_conv))]
            logger.info("Buckling spectrum: %d skipped null/gauge λ = %s",
                        n_skip, np.array2string(np.array(skipped), precision=3))

            # Extract the physical eigenpairs into normalized mode Functions.
            eigvals = np.full(n_eig, np.nan)
            n_load = min(n_eig, max(0, n_conv - n_skip))
            modes: list[fem.Function] = []
            for i in range(n_load):
                eigvals[i] = eps.getEigenvalue(n_skip + i).real
                phi = fem.Function(self.V, name=f"phi_{i}")
                vec = phi.x.petsc_vec
                eps.getEigenvector(n_skip + i, vec)
                if self.mpc is not None:
                    self.mpc.backsubstitution(vec)
                phi.x.scatter_forward()
                h1_sq = self._mesh.comm.allreduce(fem.assemble_scalar(fem.form(
                    (ufl.inner(phi, phi)
                     + ufl.inner(ufl.grad(phi), ufl.grad(phi))) * self.dx
                )), op=MPI.SUM)
                if h1_sq > 0.0:
                    vec.scale(1.0 / np.sqrt(h1_sq))
                    phi.x.scatter_forward()
                modes.append(phi)
            logger.info("Buckling spectrum (physical, smallest |λ|): %s",
                        np.array2string(eigvals, precision=4))

            if visualize_modes and modes:
                out_path = os.path.join(self.output_dir, modes_filename)
                os.makedirs(self.output_dir, exist_ok=True)
                phi_viz = fem.Function(self.V, name="phi")
                writer = io.VTXWriter(self.comm, out_path, [phi_viz], engine="BP4")
                try:
                    for i, phi in enumerate(modes):
                        phi_viz.x.array[:] = phi.x.array
                        phi_viz.x.scatter_forward()
                        writer.write(float(i))
                finally:
                    writer.close()
                logger.info("Wrote %d buckling mode(s) to %s (timestep = mode index)",
                            len(modes), out_path)

            if save_modes and modes:
                snap_dir = os.path.join(self.output_dir, "snapshots")
                if self.comm.rank == 0:
                    os.makedirs(snap_dir, exist_ok=True)
                self.comm.barrier()
                for i, phi in enumerate(modes):
                    self._save_snapshot("phi", phi, int(i))
                logger.info("Saved %d buckling mode(s) as snapshots to %s",
                            len(modes), snap_dir)

            if (save_eigvals or save_modes) and self.comm.rank == 0:
                snap_dir = os.path.join(self.output_dir, "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                np.save(os.path.join(snap_dir, "buckling_eigvals.npy"), eigvals)

            return (eigvals, modes) if return_modes else eigvals
        finally:
            eps.destroy()
            PETSc.Mat.destroy(K)

    # ------------------------------------------------------------------
    # Visualization and snapshots
    # ------------------------------------------------------------------

    def _setup_visualization(self, visualize_fields, F_var, P_ufl, J_ufl, W_ufl, u_total, output_dir) -> None:
        fields = []
        for field in visualize_fields:
            TT = fem.functionspace(self._mesh, ("DG", 1, (self.gdim, self.gdim)))
            V1 = fem.functionspace(self._mesh, ("DG", 1, (self.gdim,)))
            SS = fem.functionspace(self._mesh, ("DG", 1))
            if field == "u_fluc":
                self.u_int = fem.Function(V1, name="u_fluc")
                fields.append(self.u_int)
            elif field == "u_total":
                self.u_total = fem.Function(V1, name="u_total")
                self._u_total_expr = fem.Expression(u_total, V1.element.interpolation_points)
                fields.append(self.u_total)
            elif field == "F":
                self.F_func = fem.Function(TT, name="F")
                self._F_expr = fem.Expression(F_var, TT.element.interpolation_points)
                fields.append(self.F_func)
            elif field == "P":
                self.P_func = fem.Function(TT, name="P")
                self._P_expr = fem.Expression(P_ufl, TT.element.interpolation_points)
                fields.append(self.P_func)
            elif field == "J":
                self.J_func = fem.Function(SS, name="J")
                self._J_expr = fem.Expression(J_ufl, SS.element.interpolation_points)
                fields.append(self.J_func)
            elif field == "W":
                self.W_func = fem.Function(SS, name="W")
                self._W_expr = fem.Expression(W_ufl, SS.element.interpolation_points)
                fields.append(self.W_func)
        if fields:
            self.vtx = VTXManager(self.comm, f"{output_dir}/solution.bp", fields)
        else:
            self.vtx = None
        self.visualize_fields = visualize_fields
        logger.debug("Visualization fields: %s", visualize_fields)

    def _write_fields(self, t: float) -> None:
        if self.vtx is not None:
            for field in self.visualize_fields:
                if field == "u_fluc":
                    self.u_int.interpolate(self.u)
                elif field == "u_total":
                    self.u_total.interpolate(self._u_total_expr)
                elif field == "F":
                    self.F_func.interpolate(self._F_expr)
                elif field == "P":
                    self.P_func.interpolate(self._P_expr)
                elif field == "J":
                    self.J_func.interpolate(self._J_expr)
                elif field == "W":
                    self.W_func.interpolate(self._W_expr)
            self.vtx.write(t)
        else:
            logger.warning("No fields to write at t=%.5f (vtx is None)", t)

    def _save_snapshot(self, field_name: str, func, t_save: float) -> None:
        imap = func.function_space.dofmap.index_map
        bs = func.function_space.dofmap.index_map_bs
        n_local = imap.size_local
        owned_vals = func.x.array[:n_local * bs].copy()
        owned_coords = func.function_space.tabulate_dof_coordinates()[:n_local].copy()
        all_vals = self.comm.gather(owned_vals, root=0)
        all_coords = self.comm.gather(owned_coords, root=0)
        if self.comm.rank == 0:
            vals = np.concatenate(all_vals)
            coords = np.concatenate(all_coords, axis=0)
            snap_dir = f"{self.output_dir}/snapshots"
            np.save(f"{snap_dir}/{field_name}_{t_save:.5f}.npy", vals)
            coords_path = f"{snap_dir}/{field_name}_dof_coords.npy"
            if not os.path.exists(coords_path):
                np.save(coords_path, coords)

    # ------------------------------------------------------------------
    # Averaging and __call__
    # ------------------------------------------------------------------

    def _collect_averages(self) -> dict:
        needed: set[str] = set()
        for q in self._average_quantities:
            for name in q.required_macro_adjoints:
                needed.add(name)
        # Also include any macro variables requested through the
        # ``dw_d{name}`` snapshot path, so sensitivities exist to dump even
        # if no average quantity asked for them.
        for field in self.save_snapshots:
            if field.startswith("dw_d"):
                var_name = field[4:]
                if var_name in self._macro_var_rhs_forms:
                    needed.add(var_name)
        adjoints: dict | None = None
        if needed:
            rhs_dict = {name: self._macro_var_rhs_forms[name] for name in needed}
            adjoints = self._newton.solve_macro_sensitivities(rhs_dict)
        self._last_sensitivities = adjoints
        out: dict = {}
        for q in self._average_quantities:
            out[q.name] = q.compute(self._context, adjoints)
        return out

    def __call__(self, Fbar: np.ndarray, *,
                 pert_amplitude_init: float | None = None,
                 plot_time_start: float = 0.0) -> list[dict]:
        assert self._newton is not None, "Setup not complete."

        # Fall back to the instance default (set at construction) so the FE²
        # driver, which calls ``rve(F_qp)`` without a per-call amplitude, still
        # honours the factory-configured kick.
        if pert_amplitude_init is None:
            pert_amplitude_init = self._pert_amplitude_init

        Fbar_prev = self.F_bar_conv.copy()
        self.F_bar.value[:] = Fbar_prev
        self.u.x.array[:] = self._u_conv.x.array
        self.u.x.scatter_forward()
        self._restore_trial_state()

        target_F = np.asarray(Fbar, dtype=PETSc.ScalarType)

        # Objectivity reduction: drive the RVE with the symmetric stretch U
        # (F̄ = R U) and remember the polar data to rotate the outputs back.
        obj_R = obj_dR = obj_dU = None
        if self._objective:
            obj_R, obj_U, obj_dR, obj_dU = polar_derivatives(
                np.asarray(Fbar, dtype=float))
            target_F = np.asarray(obj_U, dtype=PETSc.ScalarType)

        def load_schedule(t: float) -> None:
            for i in range(target_F.shape[0]):
                for j in range(target_F.shape[1]):
                    self.F_bar.value[i, j] = (
                        t * (target_F[i, j] - Fbar_prev[i, j]) + Fbar_prev[i, j]
                    )
            self._update_macro_load_schedule(t)

        u = self.u
        if self.vtx is not None:
            self._write_fields(0.0 + plot_time_start)

        output_quantities: list[dict] = []
        if not self._averages_only_final:
            output_quantities.append(self._collect_averages())

        self._timestepper.reset()
        self._last_converged_Fbar = np.asarray(self.F_bar.value).copy()
        while not self._timestepper.finished:
            trial_time = self._timestepper.step_forward()
            logger.info("── Step  t=%.5f  dt=%.2e", trial_time, self._timestepper.dt)

            load_schedule(trial_time)

            stable_configuration = False
            pert_amplitude = pert_amplitude_init
            iter_newton = 0
            self._newton.reset_for_new_timestep()

            while not stable_configuration:
                converged, iter_newton = self._newton.solve(iter_start=iter_newton)

                if converged:
                    if self._stability is not None:
                        K = self._newton.assemble_stiffness()
                        try:
                            is_stable, eigenvalues = self._stability.check(K, self._eigenfunction)
                        except (PETSc.Error, SystemError):
                            logger.error("Stability check failed.")
                            ok = self._timestepper.reject()
                            if not ok:
                                logger.error(
                                    "Minimum time step dt=%.2e reached — stopping.",
                                    self._timestepper.dt_min,
                                )
                                u.x.array[:] = self._u_last.x.array[:]
                                u.x.scatter_forward()
                                raise RVEConvergenceError(
                                    f"Stability check failed and dt_min={self._timestepper.dt_min:.2e} "
                                    f"reached at t={self._timestepper.t_current:.4f}"
                                )
                            else:
                                logger.warning(
                                    "Eigensolver crashed — halving dt to %.2e",
                                    self._timestepper.dt,
                                )
                            u.x.array[:] = self._u_last.x.array[:]
                            u.x.scatter_forward()
                            break
                    else:
                        is_stable, eigenvalues = True, np.array([])

                    if is_stable:
                        stable_configuration = True
                        self._timestepper.accept(iter_newton)
                        self._u_last.x.array[:] = u.x.array[:]
                        self._u_last.x.scatter_forward()
                        self._last_converged_Fbar = np.asarray(self.F_bar.value).copy()

                        if not self._averages_only_final:
                            output_quantities.append(self._collect_averages())

                        t_save = self._timestepper.t_current + plot_time_start
                        if self.vtx is not None:
                            self._write_fields(t_save)

                        if self.save_snapshots:
                            os.makedirs(f"{self.output_dir}/snapshots", exist_ok=True)
                        for field in self.save_snapshots:
                            if field == "u_fluc":
                                self._save_snapshot("u_fluc", u, t_save)
                            elif field == "P":
                                self.P_func.interpolate(self._P_expr)
                                self._save_snapshot("P", self.P_func, t_save)
                            elif field == "A":
                                self.A_func.interpolate(self._A_expr)
                                self._save_snapshot("A", self.A_func, t_save)
                            elif field.startswith("dw_d"):
                                var_name = field[4:]
                                sens = (self._last_sensitivities or {}).get(var_name)
                                if sens is None:
                                    logger.warning(
                                        "Snapshot '%s' requested but no "
                                        "sensitivities for macro variable '%s' "
                                        "are available.", field, var_name,
                                    )
                                    continue
                                # Unravel the flat k-index into the macro
                                # variable's natural shape (e.g. Fbar (i,j),
                                # g (mode,d)) so each snapshot filename keeps
                                # its component indices.
                                mvar = self.macro_vars.get(var_name)
                                shape = (
                                    np.asarray(mvar.value).shape
                                    if mvar is not None else (len(sens),)
                                )
                                if shape == ():
                                    shape = (1,)
                                for k, p in enumerate(sens):
                                    idx = np.unravel_index(k, shape)
                                    suffix = "_".join(str(i) for i in idx)
                                    self._save_snapshot(
                                        f"dw_d{var_name}_{suffix}", p, t_save,
                                    )
                    elif self._perturb_post_buckling:
                        # Unstable: perturb onto the buckled branch and re-solve
                        # (the eigenvector kick doubles on each retry).
                        target = np.where(eigenvalues < self._stability._neg_tol)[0]
                        scale, info = apply_eigenmode_perturbation(
                            u, self._eigenfunction, pert_amplitude, self.comm,
                            char_length=self._char_length,
                        )
                        u_ref, phi_max = info[0]
                        logger.warning(
                            "Unstable equilibrium (λ_min=%.4e) — "
                            "perturbing with eigenvector "
                            "(factor=%.2e, |u|=%.2e, ‖perturbation‖_∞=%.2e)",
                            eigenvalues[target[0]], pert_amplitude, u_ref,
                            scale * phi_max,
                        )
                        pert_amplitude *= 2
                    else:
                        # Approach mode (perturb_post_buckling=False): don't jump
                        # onto the buckled branch — reject and halve dt to step as
                        # close to the bifurcation as possible.
                        logger.info(
                            "Unstable equilibrium (λ_min=%.4e) — perturbation "
                            "disabled; halving dt to approach the bifurcation",
                            float(np.min(eigenvalues)),
                        )
                        ok = self._timestepper.reject()
                        u.x.array[:] = self._u_last.x.array[:]
                        u.x.scatter_forward()
                        if not ok:
                            raise RVEConvergenceError(
                                f"Reached instability (approach mode) and "
                                f"dt_min={self._timestepper.dt_min:.2e} reached "
                                f"near t={self._timestepper.t_current:.4f}"
                            )
                        break

                else:
                    ok = self._timestepper.reject()
                    if not ok:
                        logger.error(
                            "Minimum time step dt=%.2e reached — stopping.", self._timestepper.dt_min
                        )
                        raise RVEConvergenceError(
                            f"Newton did not converge and dt_min={self._timestepper.dt_min:.2e} "
                            f"reached at t={self._timestepper.t_current:.4f}"
                        )
                    else:
                        logger.warning(
                            "Newton did not converge — halving dt to %.2e", self._timestepper.dt
                        )
                    u.x.array[:] = self._u_last.x.array[:]
                    u.x.scatter_forward()
                    break

        if self._averages_only_final:
            output_quantities.append(self._collect_averages())

        if self._objective and output_quantities:
            self._objective_transform_output(
                output_quantities[-1], obj_R, obj_dR, obj_dU)

        return output_quantities

    # ------------------------------------------------------------------
    # Objectivity reduction: U-frame → lab-frame output conversion
    # ------------------------------------------------------------------

    def _objective_transform_output(self, d: dict, R, dR, dU) -> None:
        """Convert a U-frame output dict to the lab frame *in place*.

        ``P̄ = R P̃`` and ``dP̄/dF̄ = dR·P̃ + R·(dP̃/dU)·dU`` from the U-frame
        reduced tangent (emitted under ``"dPbar_dFbar"`` by
        :class:`~fe2_rom.ch1.averages.EffectiveAbarReduced`) and the analytic
        polar derivatives. Subclasses override :meth:`_objective_transform_extra`
        for the additional micromorphic quantities.
        """
        objective_transform_pbar(d, R, dR, dU)
        self._objective_transform_extra(d, R, dR, dU)

    def _objective_transform_extra(self, d: dict, R, dR, dU) -> None:
        """Subclass hook for extra U-frame → lab-frame conversions
        (micromorphic ``Pi``/``Lambda`` blocks). Base does nothing."""
        return

    def commit(self) -> None:
        self.F_bar_conv[:] = self.F_bar.value
        self._u_conv.x.array[:] = self.u.x.array
        self._u_conv.x.scatter_forward()
        self._commit_extra_state()

    # ------------------------------------------------------------------
    # Checkpoint hooks (used by the FE² macro driver for full two-scale
    # runs; per-RVE state, one COMM_SELF mesh per qp).
    # ------------------------------------------------------------------

    def dump_state(self) -> dict[str, np.ndarray]:
        """Return the converged restart state of this RVE as a dict of
        numpy arrays (single-rank, since RVEs live on COMM_SELF)."""
        d = {
            "F_bar_conv": np.asarray(self.F_bar_conv, dtype=np.float64).copy(),
            "u_conv": np.asarray(self._u_conv.x.array, dtype=np.float64).copy(),
        }
        extra = self._dump_extra_state()
        if extra:
            d.update(extra)
        return d

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        """Load the converged restart state from a dict (inverse of
        :meth:`dump_state`). Seeds the trial state from the converged one
        so the next solve warm-starts correctly."""
        F = np.asarray(d["F_bar_conv"], dtype=PETSc.ScalarType)
        if F.shape != self.F_bar_conv.shape:
            raise RuntimeError(
                f"F_bar_conv shape mismatch: got {F.shape}, expected "
                f"{self.F_bar_conv.shape}"
            )
        u = np.asarray(d["u_conv"])
        if u.shape != self._u_conv.x.array.shape:
            raise RuntimeError(
                f"u_conv shape mismatch: got {u.shape}, expected "
                f"{self._u_conv.x.array.shape}"
            )
        self.F_bar_conv[:] = F
        self._u_conv.x.array[:] = u
        self._u_conv.x.scatter_forward()
        # Seed trial state from converged state so the next solve starts
        # at the restart point.
        self.F_bar.value[:] = self.F_bar_conv
        self.u.x.array[:] = self._u_conv.x.array
        self.u.x.scatter_forward()
        self._load_extra_state(d)

    def _dump_extra_state(self) -> dict[str, np.ndarray]:
        """Subclass hook: extra per-RVE state for checkpoint."""
        return {}

    def _load_extra_state(self, d: dict[str, np.ndarray]) -> None:
        """Subclass hook: consume extra state added by ``_dump_extra_state``."""
        return
