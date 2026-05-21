# fe2_rom

**Reduced-order FE² for hyperelastic materials in DOLFINx** — first-order and
**micromorphic** computational homogenization on periodic RVEs, with POD + ECM
hyper-reduction and full-order buckling solvers included. Built on
[FEniCSx / DOLFINx](https://fenicsproject.org/) and
[`dolfinx_mpc`](https://github.com/jorgensd/dolfinx_mpc).

The package ships with ready-to-use solvers in 2D and 3D for:

- quasi-static hyperelasticity with Newton–Raphson and arc-length continuation,
- post-buckling tracing through eigenvalue perturbation of unstable equilibria,
- **first-order computational homogenization** (CH1) on periodic RVEs
  (Hill–Mandel averaging of `F̄`, `P̄`, energy `W̄`, and tangent `Ā`),
- **micromorphic homogenization** (MM, [\[1\]](#references)) with a user-supplied enriched ansatz
  `u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w`, returning effective stresses
  `P̄`, `Πᵢ`, `Λᵢ` and the full 3 × 3 grid of tangents,
- **reduced-order modelling** (POD + ECM hyper-reduction, [\[2\]](#references)) and matching reduced
  RVE solvers for both CH1 and micromorphic kinematics,
- **two-scale FE² simulation** — a (higher-order) macroscopic continuum whose constitutive
  response at every quadrature point comes from a nested RVE, with either the
  full periodic solver or the reduced ROM as the inner driver, in both the
  classical and the micromorphic flavour.

<p align="center">
  <img src="gifs/lattice.gif" width="32%" />
  <img src="gifs/2d_rve.gif" width="32%" />
  <img src="gifs/3d_rve.gif" width="32%" />
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
- **Arc-length continuation** (`CylindricalArcLength`) for tracing snap-through
  / snap-back responses where load- or displacement-control alone fails.
- **First-order homogenization** (`fe2_rom.ch1`) on Gmsh-generated RVEs using
  `dolfinx_mpc` periodic constraints, with a modular `AverageQuantity` /
  `TangentBlock` framework for the effective quantities (`F̄`, `P̄`, `W̄`, `Ā`)
  and the matching tangents.
- **Micromorphic homogenization** (`fe2_rom.mm`) — enriched ansatz with a
  user-supplied family of global modes `φᵢ` (from a linear buckling analysis
  or any other source) and extra macro variables `(v, g=∇v)`. Reports `P̄`, `Πᵢ`,
  `Λᵢ` and the 3 × 3 grid of tangent blocks `d{P̄, Π, Λ} / d{F̄, v, g}`.
- **Projected linear constraint enforcement** — `⟨w⟩ = 0`, `⟨w·φᵢ⟩ = 0`,
  `⟨(w·φᵢ) X⟩ = 0`, etc. are imposed through a projected Newton solve
  (`P K P x = P b`); a `corner_periodic` flag toggles between fixed-corner BCs
  and full periodic constraints.
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
  - Both drivers feature adaptive load stepping, optional macro-level
    stability monitoring, and MPI parallelism over quadrature points.
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
│   ├── solvers.py          # NewtonSolver, ArcLengthSolver, CylindricalArcLength, ...
│   ├── stability.py        # SLEPc-based eigenvalue / instability analysis
│   ├── material.py         # MaterialModel, NeoHookean
│   ├── forms.py            # weak-form assembly, basis_tensor_ufl
│   ├── boundary.py         # ReactionProbe
│   ├── timestepping.py     # adaptive TimeStepper
│   └── output.py           # VTXManager, ReactionForceLogger
├── ch1/                    # first-order classical homogenization
│   ├── microsolver.py      # MicroSolver (full-order periodic RVE)
│   ├── macrosolver.py      # MacroSolver (FE² macro driver, FOM or ROM inner)
│   ├── material.py         # RVEMaterial (dolfinx_materials bridge)
│   ├── averages.py         # AverageQuantity framework: F̄, P̄, W̄, Ā, TangentBlock
│   ├── constraints.py      # LinearConstraint, ZeroVolumeAverage
│   ├── solvers.py          # NewtonSolverFE2 (projected, MPC-aware)
│   └── exceptions.py       # RVEConvergenceError
├── mm/                     # micromorphic homogenization
│   ├── microsolver.py      # MicroSolver — enriched ansatz with φ + (v, g)
│   ├── macrosolver.py      # MacroMicromorphicSolver — coupled (u, v) macro driver
│   ├── material.py         # DummyMicromorphicMaterial, MicromorphicRVEMaterial
│   ├── averages.py         # EffectivePi, EffectiveLambda
│   └── constraints.py      # ZeroVolumeAverageDot, ZeroVolumeAverageOuter
└── rom/                    # reduced-order modelling
    ├── pod.py              # POD basis construction (H¹ / L² inner products)
    ├── ecm.py              # ECM hyper-reduction (magic-point selection)
    ├── solver_ch1.py       # ReducedMicroSolver — CH1 reduced online stage
    └── solver_mm.py        # ReducedMicroSolver — micromorphic reduced online stage

examples/
├── hyperelastic_solver/
│   ├── example_1/      # 3D tetragonal lattice — compressive buckling
│   ├── example_2/      # 3D hexagonal lattice — compressive buckling
│   ├── example_3/      # 3D extruded honeycomb beam — bending/buckling
│   └── arc-length/     # snap-back of a deep parabolic arch
├── ch1/                # first-order computational homogenization
│   ├── example_1/      # 2D perforated RVE — FOM and ROM
│   ├── example_2/      # 3D periodic RVE — FOM and ROM
│   └── example_3/      # FE² — single-element macro cube × 3D periodic RVE (reuses example_2's ECM)
└── mm/                 # micromorphic homogenization
    ├── example_1/      # standalone micromorphic RVE + snapshot sampling + ROM build
    ├── example_2/      # FE² macro micromorphic with dummy constitutive law (3D)
    └── example_3/      # FE² macro micromorphic with nested RVE (FOM or ROM, 2D)
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
follows Guo et al. (2024) [\[2\]](#references).

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
buckling analysis of the RVE) and a set of macro variables `(v, g)`, following
the formulation of Rokoš et al. (2019) [\[1\]](#references):

```
u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w
```

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
)

# Populate φᵢ from a linear buckling analysis of the reference state
eigvals = solver.compute_linear_buckling_modes(N=1, save_modes=True)

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

### 6. Two-scale FE² simulation (micromorphic)

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

> Note: the FE² macro solvers (`ch1.MacroSolver`, `mm.MacroMicromorphicSolver`)
> depend on
> [`dolfinx_materials`](https://github.com/bleyerj/dolfinx_materials) — see the
> install instructions above.

## References

<a id="references"></a>

1. Rokoš, O., Ameen, M. M., Peerlings, R. H. J., & Geers, M. G. D. (2019).
   *Micromorphic computational homogenization for mechanical metamaterials with
   patterning fluctuation fields.* Journal of the Mechanics and Physics of
   Solids, **123**, 119–137.
   [doi:10.1016/j.jmps.2018.08.019](https://doi.org/10.1016/j.jmps.2018.08.019).

2. Guo, T., Rokoš, O., & Veroy, K. (2024). *A reduced order model for
   geometrically parameterized two-scale simulations of elasto-plastic
   microstructures under large deformations.* Computer Methods in Applied
   Mechanics and Engineering, **418**, 116467.
   [doi:10.1016/j.cma.2023.116467](https://doi.org/10.1016/j.cma.2023.116467).

## License

Released under the [MIT License](LICENSE) — free for academic and commercial
use; please retain the copyright notice.
