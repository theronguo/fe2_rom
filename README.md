# fe2_rom

**Reduced-order FE² for hyperelastic materials in DOLFINx** — first-order,
**micromorphic**, and **second-order (strain-gradient)** computational
homogenization on periodic RVEs, with POD + ECM hyper-reduction and full-order
buckling solvers included. Built on
[FEniCSx / DOLFINx](https://fenicsproject.org/) and
[`dolfinx_mpc`](https://github.com/jorgensd/dolfinx_mpc).

The package ships with ready-to-use solvers in 2D and 3D for:

- quasi-static hyperelasticity with Newton–Raphson,
- post-buckling tracing through eigenvalue perturbation of unstable equilibria,
- **first-order computational homogenization** (CH1) on periodic RVEs
  (Hill–Mandel averaging of `F̄`, `P̄`, energy `W̄`, and tangent `Ā`),
- **micromorphic homogenization** (MM, [\[1\]](#references), [\[2\]](#references)) with a user-supplied enriched ansatz
  `u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w`, returning effective stresses
  `P̄`, `Πᵢ`, `Λᵢ` and the full 3 × 3 grid of tangents,
- **second-order (strain-gradient) homogenization** (CH2, [\[4\]](#references)) with the
  enriched ansatz `u_total = (F̄ − I)·X + ½ X·Ḡ·X + w`, returning the effective
  stress `P̄`, the double stress `Q̄`, and the 2 × 2 grid of tangents
  `d{P̄, Q̄} / d{F̄, Ḡ}`,
- **reduced-order modelling** (POD + ECM hyper-reduction, [\[3\]](#references)) and matching reduced
  RVE solvers for both CH1 and micromorphic kinematics,
- **two-scale FE² simulation** — a (higher-order) macroscopic continuum whose constitutive
  response at every quadrature point comes from a nested RVE, with either the
  full periodic solver or the reduced ROM as the inner driver, in both the
  classical and the micromorphic flavour.

<p align="center">
  <img src="gifs/lattice.gif" width="32%" />
  <img src="gifs/2d_rve.gif" width="32%" />
  <!-- <img src="gifs/3d_rve.gif" width="32%" /> -->
  <img src="gifs/3d_10x10x10.gif" width="32%" />
</p>

<sub>Compression and buckling of various 2D/3D microstructures.</sub>

## Features

- **Hyperelastic material models** — `NeoHookean` out of the box, plus a generic
  `MaterialModel` interface to plug in any strain energy
  density.
- **Stability monitoring** — every Newton step optionally solves an
  eigenproblem (SLEPc) on the tangent stiffness. When an instability is
  detected, the lowest eigenmode is used to perturb the solution and continue
  onto the post-buckled branch.
- **Adaptive time stepping** with automatic step-size cutbacks on Newton
  failure.
- **First-order homogenization** (`fe2_rom.ch1`) on Gmsh-generated RVEs using
  `dolfinx_mpc` periodic constraints, with a modular `AverageQuantity` /
  `TangentBlock` framework for the effective quantities (`F̄`, `P̄`, `W̄`, `Ā`)
  and the matching tangents. Periodicity covers axis-aligned boxes (2D/3D) and,
  in 2D, **arbitrary polygonal unit cells** — hexagonal Wigner–Seitz or sheared
  parallelogram lattices — through lattice-vector-driven auto-pairing of
  opposite edges and vertices (`lattice_vectors=...`).
- **Micromorphic homogenization** (`fe2_rom.mm`) — enriched ansatz with a
  user-supplied family of global modes `φᵢ` (from a linear buckling analysis
  or any other source) and extra macro variables `(v, g=∇v)`. Reports `P̄`, `Πᵢ`,
  `Λᵢ` and the 3 × 3 grid of tangent blocks `d{P̄, Π, Λ} / d{F̄, v, g}` — each
  selectable by string key in `average_quantities` (e.g.
  `["Pbar", "Pi", "Lambda", "dPi_dFbar", ...]`). The
  modes are extracted from the RVE tangent by `compute_buckling_spectrum`
  (inspect the eigenvalue spectrum — any RVE, no basis needed) and loaded into
  the `φ` basis by `compute_linear_buckling_modes` / `load_buckling_modes`. With
  the objectivity reduction (`objective_reduction=True`, see below) the modes
  are treated **co-rotationally** (`φ → R φ`), making the homogenized law
  frame-indifferent — a departure from the fixed-mode ansatz of the reference
  papers.
- **Second-order (strain-gradient) homogenization** (`fe2_rom.ch2`) — extends the
  classical RVE with the second-gradient term in the ansatz
  `u_total = (F̄ − I)·X + ½ X·Ḡ·X + w`, where `Ḡ = ∇F̄` (symmetric in its last two
  indices). Reports the effective stress `P̄` and the **double stress**
  `Q̄_iJK = ⟨½(X_K P_iJ + X_J P_iK)⟩` together with the 2 × 2 grid of macro tangents
  `d{P̄, Q̄} / d{F̄, Ḡ}` (string keys `"Qbar"`, `"dPbar_dG"`, `"dQbar_dFbar"`,
  `"dQbar_dG"`). The macro continuum is the mixed `[u, F̂, L̄]` saddle-point
  formulation with the inf-sup-stable **P2-P1-P0** (triangles) / **Q2-Q1-Q0**
  (quads) element; macro buckling is detected from the saddle-point inertia
  (a stable equilibrium has exactly `m` negative eigenvalues, `m` = number of
  `L̄` multiplier DOFs). `fe2_rom.ch2` subclasses `fe2_rom.ch1` through the same
  `_setup_macro_vars` / `_build_*` hooks used by the micromorphic layer.
- **Projected linear constraint enforcement** — `⟨w⟩ = 0`, `⟨w·φᵢ⟩ = 0`,
  `⟨(w·φᵢ) X⟩ = 0`, etc. are imposed through a projected Newton solve
  (`P K P x = P b`); a `corner_periodic` flag toggles between fixed-corner BCs
  and full periodic constraints.
- **Objectivity (frame-indifference) reduction** (`fe2_rom.ch1.objectivity`) —
  material frame indifference gives `P̄(F̄) = R · P̄(U)` for the right polar
  decomposition `F̄ = R U`, so the homogenized response depends on `F̄` only
  through the symmetric stretch `U` (6 independent components in 3D, 3 in 2D).
  Passing `objective_reduction=True` to any `MicroSolver` / `ReducedMicroSolver`
  (classical *or* micromorphic, FOM *or* ROM) drives the RVE with `U` and
  reconstructs the lab-frame stress `P̄` and tangent `dP̄/dF̄` analytically from
  the polar derivatives `dR/dF̄`, `dU/dF̄` — solving only 6 (3D) / 3 (2D)
  symmetric adjoint directions instead of `gdim²`. For the micromorphic kinematics
  this realizes a **co-rotational** formulation in which the global modes
  co-rotate with the macroscopic rotation `R` (`φ → R φ`): `Π` and `Λ` come out
  rotation-invariant while `P̄` and its `v`/`g` derivatives rotate by `R`. This
  frame-indifferent treatment **differs from the fixed-mode ansatz** of Rokoš et
  al. (2019) [\[1\]](#references) / van Bree et al. (2020)
  [\[2\]](#references), where the modes do not co-rotate. Polar derivatives use the
  `C = U²` square-root route, which stays non-singular for repeated stretches
  (including `F̄ = I`). For ROM training, `sample_symmetric_stretch=True` in
  `generate_training_data` samples only `U` — rotation-free snapshots over a
  6-dim (3D) / 3-dim (2D) box instead of the full 9.
- **ROM toolkit** (`fe2_rom.rom`) — POD basis construction with `H¹` / `L²`
  inner products, ECM hyper-reduction (magic-point selection), and reduced
  online solvers for both CH1 (`ReducedMicroSolver`) and micromorphic
  kinematics.
- **FE² macro solvers** —
  - `fe2_rom.ch1.MacroSolver` wires a macroscopic continuum to a nested
    first-order RVE at every quadrature point through `dolfinx_materials`.
    A single `full=True/False` flag switches the inner driver between the
    full periodic `MicroSolver` and the ECM-reduced `ReducedMicroSolver`.
  - `fe2_rom.mm.MacroMicromorphicSolver` couples the macroscopic
    displacement field `u` with `N_modes` scalar enrichment-amplitude fields
    `vᵢ`, with the constitutive response provided either by a `MicromorphicRVEMaterial`
    (nested RVE, FOM or ROM).
  - `fe2_rom.ch2.MacroSecondOrderSolver` solves the mixed `[u, F̂, L̄]`
    second-order continuum (displacement, deformation gradient, and Lagrange
    multiplier), with the constitutive response from a `Ch2RVEMaterial`
    (nested second-order RVE, FOM or ROM) or a `DummyCh2Material` for verifying
    the formulation.
  - All drivers feature adaptive load stepping, optional macro-level
    stability monitoring, checkpoint/restart, and MPI parallelism over
    quadrature points.
- **Neural-network effective-energy surrogate** (`fe2_rom.nn`) — a single
  JAX / [equinox](https://github.com/patrick-kidger/equinox) `EnergyNet` learns
  the homogenized strain energy `W̄` and replaces the nested RVE solve with a
  closed-form macro law, collapsing the two-scale problem to a single
  macroscopic one. One `flavor` switch covers all three homogenization layers
  (`"ch1"`: `W̄(F̄)`, `"mm"`: `W̄(F̄, v, g)`, `"ch2"`: `W̄(F̄, Ḡ)`); every stress
  and tangent is an automatic derivative of one scalar, so frame indifference
  (objectivity via the polar decomposition), the micromorphic sign symmetry
  `W̄(U, −v, −g) = W̄(U, v, g)`, and the exact zero-energy/zero-stress reference
  state hold **by construction**. Trained with a Sobolev loss (energy + stresses)
  via [optax](https://github.com/google-deepmind/optax) (`fe2_rom.nn.training`).
  Deployed through drop-in `dolfinx_materials` laws (`NNRVEMaterial`,
  `NNMicromorphicMaterial`, `NNCh2Material`) that plug straight into the macro
  drivers above. Custom architectures (e.g. a constitutive-ANN) subclass via the
  `_build_core` hook.
- **Differentiable weight fitting** (`fe2_rom.fit.MacroFit`) — fit the
  `EnergyNet` weights to a macro observation (e.g. a DNS reaction curve):
  `fit(θ)` runs the macro solve and returns the reaction at each load level,
  `fit.jac(θ)` returns `d(reaction)/dθ` through a per-level discrete adjoint
  (JAX vector–Jacobian product for the flux–weight term), ready for
  `scipy.optimize.least_squares`.
- **VTX (ADIOS2) output** for ParaView visualisation and CSV reaction-force
  logging.

## Installation

A conda/mamba environment file is provided:

```bash
mamba env create -f environment.yml
mamba activate fe2_rom_env
pip install -e .
```

The FE² macro solvers (`fe2_rom.ch1.MacroSolver`,
`fe2_rom.mm.MacroMicromorphicSolver`) additionally require
[`dolfinx_materials`](https://github.com/bleyerj/dolfinx_materials), which is
not on PyPI. Clone it alongside this repo and install:

```bash
git clone https://github.com/bleyerj/dolfinx_materials
pip install dolfinx_materials/ --user
```

A `Dockerfile` and `Singularity.def` are also included for containerised
deployments (HPC, CI, reproducible runs).

## Repository layout

```
fe2_rom/
├── hyperelastic_solver/    # full-order FE solver
│   ├── solver.py           # HyperelasticStabilitySolver
│   ├── solvers.py          # NewtonSolver, ...
│   ├── stability.py        # SLEPc-based eigenvalue / instability analysis
│   ├── material.py         # MaterialModel, NeoHookean
│   ├── forms.py            # weak-form assembly, basis_tensor_ufl
│   ├── boundary.py         # ReactionProbe
│   ├── timestepping.py     # adaptive TimeStepper
│   └── output.py           # VTXManager, ReactionForceLogger
├── ch1/                    # first-order classical homogenization
│   ├── microsolver.py      # MicroSolver (full-order periodic RVE; box or 2D polygon)
│   ├── macrosolver.py      # MacroSolver (FE² macro driver, FOM or ROM inner)
│   ├── material.py         # RVEMaterial (dolfinx_materials bridge)
│   ├── averages.py         # AverageQuantity framework: F̄, P̄, W̄, Ā, TangentBlock
│   ├── objectivity.py      # frame-indifference reduction (F̄ = R U): drive RVE with U
│   ├── constraints.py      # LinearConstraint, ZeroVolumeAverage
│   ├── solvers.py          # NewtonSolverFE2 (projected, MPC-aware)
│   └── exceptions.py       # RVEConvergenceError
├── mm/                     # micromorphic homogenization
│   ├── microsolver.py      # MicroSolver — enriched ansatz with φ + (v, g)
│   ├── macrosolver.py      # MacroMicromorphicSolver — coupled (u, v) macro driver
│   ├── material.py         # DummyMicromorphicMaterial, MicromorphicRVEMaterial
│   ├── averages.py         # EffectivePi, EffectiveLambda
│   └── constraints.py      # ZeroVolumeAverageDot, ZeroVolumeAverageOuter
├── ch2/                    # second-order (strain-gradient) homogenization
│   ├── microsolver.py      # MicroSolver — adds the ½ X·Ḡ·X second-gradient term
│   ├── macrosolver.py      # MacroSecondOrderSolver — mixed [u, F̂, L̄] macro driver
│   ├── material.py         # Ch2RVEMaterial, DummyCh2Material
│   ├── averages.py         # EffectiveQ (double stress Q̄) + x-weighted tangent blocks
│   ├── constraints.py      # ZeroBoundaryAverage
│   └── training_data.py    # (F̄, Ḡ) snapshot generation for the ROM
├── rom/                    # reduced-order modelling
│   ├── pod.py              # POD basis construction (H¹ / L² inner products)
│   ├── ecm.py              # ECM hyper-reduction (magic-point selection)
│   ├── solver_ch1.py       # ReducedMicroSolver — CH1 reduced online stage
│   ├── solver_mm.py        # ReducedMicroSolver — micromorphic reduced online stage
│   └── solver_ch2.py       # ReducedMicroSolver — second-order reduced online stage
├── nn/                     # JAX/equinox effective-energy surrogate
│   ├── model.py            # EnergyNet (flavor ch1/mm/ch2), make_lab_energy
│   ├── polar.py            # differentiable polar decomposition (Newton–Schulz)
│   ├── training.py         # optax Sobolev training (train_energy)
│   └── sensitivity.py      # weight VJP + params<->vector (for MacroFit)
└── fit/                    # differentiable weight fitting
    └── macrofit.py         # MacroFit: fit(θ)->reaction, fit.jac via adjoint

# NN materials live next to their homogenization layer:
#   ch1/nn_material.py (NNRVEMaterial), mm/nn_material.py
#   (NNMicromorphicMaterial), ch2/nn_material.py (NNCh2Material)

examples/
├── hyperelastic_solver/
│   ├── example_1/      # 3D tetragonal lattice — compressive buckling
│   ├── example_2/      # 3D hexagonal lattice — compressive buckling
│   └── example_3/      # 3D extruded honeycomb beam — bending/buckling
├── ch1/                # first-order computational homogenization
│   ├── example_1/      # 2D perforated RVE — FOM and ROM
│   ├── example_2/      # 3D periodic RVE — FOM and ROM
│   └── example_3/      # FE² — single-element macro cube × 3D periodic RVE (reuses example_2's ECM)
└── mm/                 # micromorphic homogenization
    ├── example_1/      # standalone micromorphic RVE + snapshot sampling + ROM build
    ├── example_2/      # FE² macro micromorphic with dummy constitutive law (3D)
    ├── example_3/      # FE² macro micromorphic with nested RVE (FOM or ROM, 2D)
    └── example_4/      # polygonal (hexagonal) RVE — buckling spectrum + micromorphic solve
```

## Quick start

### 1. Hyperelastic buckling (3D lattice)

```python
from mpi4py import MPI
from dolfinx import fem, io
from petsc4py import PETSc
from fe2_rom.hyperelastic_solver import (
    HyperelasticStabilitySolver, NeoHookean,
    TimeStepper, VTXManager, ReactionForceLogger,
)

comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags, *_ = io.gmsh.read_from_msh("lattice.msh", comm, 0, gdim=3)

material = NeoHookean(mu=1000.0, lmbda=2000.0)
solver = HyperelasticStabilitySolver(mesh, cell_tags, facet_tags, material)

u_top = fem.Constant(mesh, PETSc.ScalarType(0.0))
solver.add_bc(2, lambda x: x[2] < 1e-8, fem.Constant(mesh, 0.0))
solver.add_bc(2, lambda x: x[2] > 1 - 1e-8, u_top,
              measure_reaction=True, reaction_direction=(0.0, 0.0, 1.0))
solver.setup(check_stability=True)

solver.run(
    load_schedule=lambda t: setattr(u_top, "value", -0.25 * t),
    timestepper=TimeStepper(t_end=1.0, dt_init=0.1, dt_min=1e-5),
    output_manager=VTXManager(comm, "out.bp",
        [solver.u_int, solver.F_func, solver.P_func, solver.J_func]),
    reaction_logger=ReactionForceLogger(),
    pert_amplitude_init=1e1,   # eigenmode perturbation at the first bifurcation
)
```

Run the full example:

```bash
cd examples/hyperelastic_solver/example_1
python run_solver.py            # serial
mpirun -n 4 python run_solver.py  # parallel
```

### 2. First-order computational homogenization (CH1)

```python
import numpy as np
from mpi4py import MPI
from fe2_rom.hyperelastic_solver import NeoHookean
from fe2_rom.ch1 import MicroSolver

solver = MicroSolver(
    mesh_path="rve.msh", comm=MPI.COMM_WORLD, gdim=2,
    material=NeoHookean(mu=1153.8, lmbda=1730.8),
    average_quantities=["F", "P", "A"],    # F̄, P̄, Ā via the modular framework
    save_snapshots=["u_fluc", "P"],        # for later POD
    check_stability=True,
)

# Apply a macroscopic deformation gradient F̄
F_bar = np.array([[0.8, 0.0], [0.0, 1.0]])
history = solver(F_bar, pert_amplitude_init=1e-1)
# history[i] is a dict per accepted load step with keys "Fbar", "Pbar",
# "dPbar_dFbar", ...
```

Run the full example:

```bash
cd examples/ch1/example_1
python run_homogenization.py
```

### 3. Build a ROM and run the reduced CH1 solver

```bash
cd examples/ch1/example_1
python run_homogenization.py        # generate snapshots in output/
python build_rom.py                 # POD + ECM → ecm/
python run_homogenization_rom.py    # reduced online stage
```

`build_rom.py` constructs POD bases (energy criterion 99.99%) for the
fluctuation displacement `u_fluc` (`H¹` inner product) and first
Piola–Kirchhoff stress `P` (`L²`), then ECM hyper-reduction selects "magic
points" on a sub-mesh. The reduced solver
(`fe2_rom.rom.ReducedMicroSolver`) reproduces `P̄(F̄)` and the tangent
`Ā(F̄)` at a fraction of the full-order cost. The POD + ECM construction
follows Guo et al. (2024) [\[3\]](#references).

### 4. Two-scale FE² simulation (classical, CH1)

`fe2_rom.ch1.MacroSolver` wires a macroscopic continuum to a nested RVE at
every macro quadrature point. The inner driver is selected by a single
`full` flag — set `full=True` to use the full periodic `MicroSolver`, or
`full=False` (with `rom_dir=...`) to drive each qp with the ECM-reduced
`ReducedMicroSolver`.

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from fe2_rom.hyperelastic_solver import NeoHookean, TimeStepper, ReactionForceLogger
from fe2_rom.ch1 import MacroSolver

comm = MPI.COMM_WORLD
domain = dmesh.create_unit_cube(comm, 1, 1, 1, cell_type=dmesh.CellType.hexahedron)

solver = MacroSolver(
    mesh=domain,
    full=True,                              # full=False + rom_dir=... for ROM-backed FE²
    n_qp=1,
    rve_mesh_path="mesh.msh",
    rve_material=NeoHookean(mu=1153.8, lmbda=1730.8),
    rve_average_quantities=["P", "A"],       # MacroSolver reads Pbar, dPbar_dFbar
    rve_check_stability=True,
    gdim=3, degree=1,
    check_stability=True,                    # macro-level instability monitoring
)

zero, disp = fem.Constant(domain, 0.0), fem.Constant(domain, 0.0)
for sub in (0, 1, 2):
    solver.add_bc(sub, lambda x: np.isclose(x[2], 0.0), zero)
solver.add_bc(2, lambda x: np.isclose(x[2], 1.0), disp,
              measure_reaction=True, reaction_direction=(0.0, 0.0, 1.0))
solver.setup()

solver.solve(
    output_dir="output",
    timestepper=TimeStepper(t_end=1.0, dt_init=1.0, dt_min=1e-3),
    loadhistory=lambda t: setattr(disp, "value", -0.05 * t),
    reaction_logger=ReactionForceLogger(),
)
```

Run the full example (single hex element, 3D RVE inside):

```bash
cd examples/ch1/example_3
python run_macro.py
```

The example runs with `full=True` (FOM RVE) out of the box. To switch the
inner driver to the ROM (`full=False`), first build the ECM artifacts in
`examples/ch1/example_2` (`python build_rom.py`) — `run_macro.py` reads
`rom_dir=../example_2/ecm` so the ROM is shared with that example rather
than duplicated on disk.

Each accepted macro step commits every RVE's state so nested solves
warm-start from their previous converged configuration. On RVE non-convergence
the macro step is rejected and the timestepper halves dt; on macro
instability the displacement is perturbed along the lowest eigenmode and the
Newton solve is re-driven.

### 5. Micromorphic RVE homogenization

The micromorphic `MicroSolver` enriches the classical periodic ansatz with a
user-supplied family of global modes `φᵢ` (extracted here from a linear
buckling analysis of the RVE) and a set of macro variables `(v, g)`, building on
the formulation of Rokoš et al. (2019) [\[1\]](#references):

```
u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w
```

With the objectivity reduction enabled (`objective_reduction=True`) the modes
**co-rotate** with the macroscopic rotation `R` from `F̄ = R U`, i.e. the
enrichment uses `R φᵢ` in the lab frame:

```
u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) (R φᵢ) + w
```

This co-rotational treatment makes the homogenized law frame-indifferent and
**differs from the fixed-mode ansatz** of Rokoš et al. (2019) / van Bree et al.
(2020) [\[2\]](#references). See feature notes above and `fe2_rom.ch1.objectivity`.

```python
import numpy as np
from mpi4py import MPI
from fe2_rom.hyperelastic_solver import NeoHookean
from fe2_rom.mm import MicroSolver

solver = MicroSolver(
    mesh_path="rve.msh", comm=MPI.COMM_WORLD, gdim=2,
    material=NeoHookean(mu=1153.8, lmbda=1730.8),
    N=1,                       # number of enrichment modes φᵢ
    degree=2,
    save_snapshots=["u_fluc", "P"],
    check_stability=True,
    # objective_reduction=True,  # co-rotational (φ → R φ), drive RVE with U
)

# Populate φᵢ from a linear buckling analysis of the reference state
eigvals = solver.compute_linear_buckling_modes(1, save_modes=True)

# Apply macro inputs (F̄, v, g)
Fbar = np.array([[0.9, 0.0], [0.0, 1.0]])
v    = np.array([1.0])
g    = np.array([[-0.2, 0.1]])

history = solver(Fbar, v, g, pert_amplitude_init=0.1)
# history[i] keys: Fbar, Pbar, Pi, Lambda, plus the 3×3 grid of tangents
# dPbar_dFbar, dPbar_dv, dPbar_dg, dPi_dFbar, dPi_dv, dPi_dg,
# dLambda_dFbar, dLambda_dv, dLambda_dg
```

Run the standalone example and build a micromorphic ROM:

```bash
cd examples/mm/example_1
python run_micromorphic.py        # FD-verified standalone solve, saves φ snapshots
python sample_micromorphic.py     # sample (F̄, v, g) space for ROM training
python build_rom.py               # POD + ECM on the micromorphic snapshots
python run_micromorphic_rom.py    # reduced micromorphic online stage
```

### 6. Polygonal RVE periodicity & automatic buckling-mode selection (hexagonal)

Periodicity is not restricted to axis-aligned boxes. In 2D, supplying the two
lattice vectors makes the (classical *or* micromorphic) `MicroSolver` auto-pair
opposite edges and vertices of an **arbitrary polygonal unit cell** — a
hexagonal Wigner–Seitz cell, a sheared parallelogram, … — through `dolfinx_mpc`
geometric ties. The cell area is passed via `rve_volume`; the mesh must have
node-for-node matching on opposite edges (e.g. Gmsh `setPeriodic`).

```python
import numpy as np
from mpi4py import MPI
from fe2_rom.hyperelastic_solver import NeoHookean
from fe2_rom.mm import MicroSolver

ell = 1.0                                            # hexagon apothem
lattice = np.array([[2 * ell, 0.0],                  # opposite-edge translations
                    [ell, np.sqrt(3) * ell]])

solver = MicroSolver(
    mesh_path="hexagonal_rve.msh", comm=MPI.COMM_WORLD, gdim=2,
    material=NeoHookean(mu=1153.8, lmbda=1730.8),
    N=0,                                             # no φ basis yet — spectrum only
    lattice_vectors=lattice,                         # polygonal periodicity
    rve_volume=2 * np.sqrt(3) * ell**2,              # exact cell area |Q|
)

# Inspect the linear buckling spectrum of the RVE tangent (works with N=0)
eigvals = solver.compute_buckling_spectrum(8, visualize_modes=True)
```

`compute_buckling_spectrum` returns the tangent's eigenvalues without touching
any mode basis, so the number of micromorphic modes can be chosen from the
spectral gap *before* building the basis. The micromorphic
`compute_linear_buckling_modes` then populates `self._phi` with the softest
modes (and `load_buckling_modes` loads a pre-computed set).

`examples/mm/example_4` runs the whole chain on a porous hexagonal RVE —
generate the mesh, inspect the spectrum, auto-select the mode count from the
spectral gap, populate the `φ` basis, and run a micromorphic solve:

```bash
cd examples/mm/example_4
python generate_hexagonal_mesh.py     # periodic (D6-symmetric) hexagonal RVE
python run_micromorphic.py            # spectrum → auto-select N → populate φ → micromorphic solve
```

### 7. Two-scale FE² simulation (micromorphic)

`fe2_rom.mm.MacroMicromorphicSolver` couples the macroscopic displacement
field `u` with `N_modes` scalar enrichment-amplitude fields `vᵢ` through the
weak form

```
∫ P : ∇(δu) dX = 0                                      (u-block)
∫ Πᵢ δvᵢ dX + ∫ Λᵢ · ∇(δvᵢ) dX = 0   ∀ i                (v-block)
```

The constitutive response at each macro qp can be supplied by either a
`DummyMicromorphicMaterial` (linear, decoupled — useful for verifying the
mixed formulation) or a `MicromorphicRVEMaterial` (nested micromorphic RVE,
FOM or ROM).

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_square, CellType

from fe2_rom.hyperelastic_solver import NeoHookean, TimeStepper
from fe2_rom.mm import MacroMicromorphicSolver, MicromorphicRVEMaterial
from fe2_rom.rom import ReducedMicroSolver

phi_arrays = [np.load("phi_0.npy")]    # produced by run_micromorphic.py
N_MODES = len(phi_arrays)

def rve_factory(rank, index):
    return ReducedMicroSolver(
        mesh_path="rve.msh", rom_dir="ecm",
        material=NeoHookean(mu=1153.8, lmbda=1730.8),
        phi=phi_arrays, gdim=2, degree=2,
        comm=MPI.COMM_SELF,
    )

material = MicromorphicRVEMaterial(rve_factory, N_modes=N_MODES, gdim=2)

mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)
solver = MacroMicromorphicSolver(mesh, n_qp=2, N_modes=N_MODES, material=material)

zero, disp = fem.Constant(mesh, 0.0), fem.Constant(mesh, 0.0)
solver.add_bc((0, 0), lambda x: np.isclose(x[0], 0.0), zero)
solver.add_bc((0, 1), lambda x: np.isclose(x[0], 0.0), zero)
solver.add_bc((0, 0), lambda x: np.isclose(x[0], 1.0), disp,
              measure_reaction=True)
for i in range(N_MODES):
    solver.add_bc((i + 1,), lambda x: np.ones(x.shape[1], dtype=bool), zero)
solver.setup()

solver.solve(
    output_dir="output",
    timestepper=TimeStepper(t_end=1.0, dt_init=0.25, dt_min=1e-6),
    loadhistory=lambda t: setattr(disp, "value", -0.02 * t),
)
```

Run the full examples:

```bash
cd examples/mm/example_2
python run_macro_micromorphic.py    # 3D macro cube, dummy material — verifies formulation

cd ../example_3
python run_macro_micromorphic.py    # 2D macro × micromorphic RVE (FOM or ROM)
```

Set `USE_ROM = True` in `example_3/run_macro_micromorphic.py` to swap the
nested FOM `MicroSolver` for the reduced `ReducedMicroSolver`. The micromorphic
example_3 expects `phi_*.npy` and the `ecm/` directory to already exist in
`example_1/output/snapshots/` and `example_1/ecm/` respectively — run
`example_1` first.

> Note: the FE² macro solvers (`ch1.MacroSolver`, `mm.MacroMicromorphicSolver`,
> `ch2.MacroSecondOrderSolver`) depend on
> [`dolfinx_materials`](https://github.com/bleyerj/dolfinx_materials) — see the
> install instructions above.

### 8. Second-order (strain-gradient) homogenization (CH2)

The second-order `MicroSolver` adds the second-gradient term `½ X·Ḡ·X` to the
classical periodic ansatz and reports the double stress `Q̄` alongside `P̄`
(see Kouznetsova et al. (2002) [\[4\]](#references)):

```
u_total = (F̄ − I)·X + ½ X·Ḡ·X + w        Ḡ_iJK = ∂F̄_iJ/∂X_K  (symmetric in J↔K)
```

```python
import numpy as np
from mpi4py import MPI
from fe2_rom.hyperelastic_solver import NeoHookean
from fe2_rom.ch2 import MicroSolver

solver = MicroSolver(
    mesh_path="rve.msh", comm=MPI.COMM_WORLD, gdim=2,
    material=NeoHookean(mu=1153.8, lmbda=1730.8),
    average_quantities=["Fbar", "Pbar", "Qbar",
                        "dPbar_dFbar", "dPbar_dG", "dQbar_dFbar", "dQbar_dG"],
    degree=2, check_stability=True,
)

Fbar = np.array([[0.9, 0.0], [0.0, 1.0]])
G = np.zeros((2, 2, 2))            # Ḡ_iJK = ∂F̄_iJ/∂X_K
G[1, 1, 1] = 0.01
history = solver(Fbar, G, pert_amplitude_init=0.1)
# history[i] keys: Fbar, Pbar, Qbar, dPbar_dFbar, dPbar_dG, dQbar_dFbar, dQbar_dG
```

The macro driver `MacroSecondOrderSolver` solves the mixed `[u, F̂, L̄]`
saddle-point problem (displacement, deformation gradient, Lagrange multiplier)
on the inf-sup-stable **P2-P1-P0** (triangles) / **Q2-Q1-Q0** (quads) element,
with the constitutive response from a nested second-order RVE (`Ch2RVEMaterial`,
FOM or ROM) or a `DummyCh2Material` for verifying the formulation:

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_rectangle, CellType, GhostMode
from fe2_rom.hyperelastic_solver import NeoHookean, TimeStepper
from fe2_rom.ch2 import MacroSecondOrderSolver, Ch2RVEMaterial
from fe2_rom.ch2.microsolver import MicroSolver

def rve_factory(rank, index):
    return MicroSolver(mesh_path="rve.msh", comm=MPI.COMM_SELF, gdim=2,
                       material=NeoHookean(mu=1153.8, lmbda=1730.8), degree=2)

material = Ch2RVEMaterial(rve_factory, gdim=2)
mesh = create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 4.0]], [1, 4],
                        CellType.triangle, ghost_mode=GhostMode.shared_facet)
solver = MacroSecondOrderSolver(mesh, n_qp=2, material=material, degree=1,
                                check_stability=True)  # P2-P1-P0; macro buckling via saddle inertia

# BC signature: an int component targets a displacement component u_i (subspace 0);
# a tuple (1, c) targets flattened component c (row-major i*gdim+j) of H = F̂ − I
# (value 0 pins F̂_ij to δ_ij); (2, c) targets the multiplier L̄.
zero, disp = fem.Constant(mesh, 0.0), fem.Constant(mesh, 0.0)
solver.add_bc(0, lambda x: np.isclose(x[1], 0.0), zero)              # u_x
solver.add_bc(1, lambda x: np.isclose(x[1], 0.0), zero)              # u_y
solver.add_bc(1, lambda x: np.isclose(x[1], 4.0), disp, measure_reaction=True)
solver.setup()

solver.solve(
    output_dir="output",
    timestepper=TimeStepper(t_end=1.0, dt_init=0.05, dt_min=1e-6),
    loadhistory=lambda t: setattr(disp, "value", -0.2 * t),
    pert_amplitude_init=1e-2,    # eigenmode kick applied when macro buckling is detected
)
```

## Validation

The micromorphic two-scale solver is validated against the benchmark of
van Bree et al. (2020) [\[2\]](#references) — the local-vs-global buckling of a
pattern-transforming metamaterial column (their Section 4.1). The suite under
`validation/mm_2d/example_1_local_global/` reproduces that example on a common
microstructure (a 2ℓ × 2ℓ RVE with four circular holes) and macroscopic specimen
(W = 4ℓ × H = 8ℓ) under compression, with three drivers:

- `run_macro_dns.py` — a fully-resolved **direct numerical simulation** (DNS)
  reference;
- `run_macro_ch1.py` — **first-order** FE² (classical homogenization), the
  baseline, which deviates from the DNS by up to ~40% in the post-buckling
  regime;
- `run_macro_mm.py` — the **micromorphic** FE², which captures the
  local→global buckling transition to within ~6% of the DNS in applied strain.

All three output the nominal stress `P₂₂` versus compressive strain `u/H`, so the
curves are directly comparable.

```bash
cd validation/mm_2d/example_1_local_global
python create_dns_mesh.py        # writes dns_6x30.msh (rve.msh ships with the repo)
python run_macro_dns.py          # DNS reference
python run_macro_ch1.py          # first-order FE² baseline
python run_macro_mm.py           # micromorphic FE²
python run_macro_mm.py --objective  # co-rotational (φ → R φ) objectivity reduction
```

The reference paper is bundled at
[`validation/mm_2d/paper/reference.pdf`](validation/mm_2d/paper/reference.pdf)
(open access, CC BY 4.0) — DOI
[10.1016/j.cma.2020.113333](https://doi.org/10.1016/j.cma.2020.113333).

## References

<a id="references"></a>

1. Rokoš, O., Ameen, M. M., Peerlings, R. H. J., & Geers, M. G. D. (2019).
   *Micromorphic computational homogenization for mechanical metamaterials with
   patterning fluctuation fields.* Journal of the Mechanics and Physics of
   Solids, **123**, 119–137.
   [doi:10.1016/j.jmps.2018.08.019](https://doi.org/10.1016/j.jmps.2018.08.019).

2. van Bree, S. E. H. M., Rokoš, O., Peerlings, R. H. J., Doškář, M., &
   Geers, M. G. D. (2020). *A Newton solver for micromorphic computational
   homogenization enabling multiscale buckling analysis of pattern-transforming
   metamaterials.* Computer Methods in Applied Mechanics and Engineering,
   **372**, 113333.
   [doi:10.1016/j.cma.2020.113333](https://doi.org/10.1016/j.cma.2020.113333).
   Open access (CC BY 4.0); a copy is bundled at
   [`validation/mm_2d/paper/reference.pdf`](validation/mm_2d/paper/reference.pdf).

3. Guo, T., Rokoš, O., & Veroy, K. (2024). *A reduced order model for
   geometrically parameterized two-scale simulations of elasto-plastic
   microstructures under large deformations.* Computer Methods in Applied
   Mechanics and Engineering, **418**, 116467.
   [doi:10.1016/j.cma.2023.116467](https://doi.org/10.1016/j.cma.2023.116467).

4. Kouznetsova, V. G., Geers, M. G. D., & Brekelmans, W. A. M. (2002).
   *Multi-scale constitutive modelling of heterogeneous materials with a
   gradient-enhanced computational homogenization scheme.* International Journal
   for Numerical Methods in Engineering, **54**(8), 1235–1260.
   [doi:10.1002/nme.541](https://doi.org/10.1002/nme.541).

## Contact

Questions, issues, or collaboration ideas? Contact **Theron Guo** —
[t.guo@tue.nl](mailto:t.guo@tue.nl).

## License

Released under the [MIT License](LICENSE) — free for academic and commercial
use; please retain the copyright notice.
