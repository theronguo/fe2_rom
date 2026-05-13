# fe2_rom

**Reduced-order FE² for hyperelastic materials in DOLFINx** — POD + ECM
hyper-reduction on periodic RVEs, with full-order buckling and homogenization
solvers included. Built on [FEniCSx / DOLFINx](https://fenicsproject.org/) and
[`dolfinx_mpc`](https://github.com/jorgensd/dolfinx_mpc).

The package ships with ready-to-use solvers in 2D and 3D for:

- quasi-static hyperelasticity with Newton–Raphson and arc-length continuation,
- post-buckling tracing through eigenvalue perturbation of unstable equilibria,
- first-order **computational homogenization** on periodic RVEs (Hill–Mandel
  averaging of `F`, `P`, energy `W`, and tangent `A`),
- **reduced-order modelling** (POD + ECM hyper-reduction) and a matching
  reduced RVE solver for fast online queries.

<p align="center">
  <img src="gifs/lattice.gif" width="32%" />
  <img src="gifs/2d_rve.gif" width="32%" />
  <img src="gifs/3d_rve.gif" width="32%" />
</p>

<sub>Left: buckling of a 3D lattice under compression. Middle/right: 2D and 3D
periodic RVEs under macroscopic stretch.</sub>

## Features

- **Hyperelastic material models** — `NeoHookean` out of the box, plus a generic
  `MaterialModel` / `LambdaMaterial` interface to plug in any stored-energy
  density.
- **Stability monitoring** — every Newton step optionally solves a generalised
  eigenproblem (SLEPc) on the tangent stiffness. When an instability is
  detected, the lowest eigenmode is used to perturb the solution and continue
  onto the post-buckled branch.
- **Adaptive time stepping** with automatic step-size cutbacks on Newton
  failure.
- **Arc-length continuation** (`CylindricalArcLength`) for tracing snap-through
  / snap-back responses where load- or displacement-control alone fails.
- **Periodic homogenization** on Gmsh-generated RVEs using `dolfinx_mpc`
  periodic constraints, with macroscopic-gradient driving and effective
  quantities `F̄, P̄, W̄, Ā` exported on demand.
- **ROM toolkit** (`fe2_rom.rve_rom`) — POD basis construction with `H¹` / `L²`
  inner products, ECM hyper-reduction (magic-point selection), and an
  `RVESolver` that runs entirely on the reduced submesh.
- **MPI-parallel** snapshot generation, ROM assembly, and online evaluation.
- **VTX (ADIOS2) output** for ParaView visualisation and CSV reaction-force
  logging.

## Installation

A conda/mamba environment file is provided:

```bash
mamba env create -f environment.yml
mamba activate fe2_rom_env
pip install -e .
```

The FE² macro solver (`fe2_rom.macro_solver`) additionally requires
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
│   ├── solver.py           # HyperelasticStabilitySolver, PeriodicHyperelasticHomogenizationSolver
│   ├── solvers.py          # NewtonSolver, ArcLengthSolver, CylindricalArcLength, ...
│   ├── stability.py        # SLEPc-based eigenvalue / instability analysis
│   ├── material.py         # MaterialModel, NeoHookean, LambdaMaterial
│   ├── forms.py            # weak-form assembly
│   ├── boundary.py         # ReactionProbe
│   ├── timestepping.py     # adaptive TimeStepper
│   └── output.py           # VTXManager, ReactionForceLogger
└── rve_rom/                # reduced-order modelling
    ├── pod.py              # POD basis, ECM hyper-reduction
    └── solver.py           # RVESolver (reduced online stage)

examples/
├── hyperelastic_solver/
│   ├── example_1/      # 3D tetragonal lattice — compressive buckling
│   ├── example_2/      # 3D hexagonal lattice — compressive buckling
│   ├── example_3/      # 3D extruded honeycomb beam — bending/buckling
│   └── arc-length/     # snap-back of a deep parabolic arch
└── periodic_solver/
    ├── example_1/      # 2D perforated RVE — full-order and ROM
    └── example_2/      # 3D periodic RVE — full-order and ROM
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

### 2. Snap-back with arc-length continuation

```bash
cd examples/hyperelastic_solver/arc-length
python run_snap_back.py
```

This traces the full (crown-displacement, load-factor) curve of a deep
parabolic arch, where both `λ` and the displacement reverse simultaneously.

### 3. Periodic RVE homogenization

```python
from fe2_rom.hyperelastic_solver import PeriodicHyperelasticHomogenizationSolver, NeoHookean
import numpy as np

solver = PeriodicHyperelasticHomogenizationSolver(
    mesh_path="rve.msh", comm=MPI.COMM_WORLD, gdim=2,
    material=NeoHookean(mu=1153.8, lmbda=1730.8),
    average_fields=["F", "P", "A"],
    save_snapshots=["u_fluc", "P"],   # for later POD
    check_stability=True,
)

# Apply a macroscopic deformation gradient F̄
F_bar = np.array([[0.8, 0.0], [0.0, 1.0]])
history = solver(F_bar, pert_amplitude_init=1e-1)
# history is a list of (F̄, P̄, Ā) tuples along the load path
```

### 4. Build a ROM and run the reduced solver

```bash
cd examples/periodic_solver/example_1
python run_homogenization.py    # generates snapshots in output/
python build_rom.py             # POD + ECM → ecm/
python run_homogenization_rom.py
```

`build_rom.py` constructs POD bases (energy criterion 99.99%), then ECM
hyper-reduction selects "magic points" on a sub-mesh. The reduced solver
(`fe2_rom.rve_rom.solver.RVESolver`) reproduces `P̄(F̄)` and the tangent `Ā(F̄)` at a
fraction of the full-order cost.

## License

Released under the [MIT License](LICENSE) — free for academic and commercial
use; please retain the copyright notice.
