"""Stage 2: micromorphic ROM training-data generation for the hexagonal RVE.

Analogue of examples/mm/example_1/generate_training_data.py for the porous
*hexagonal* cell. The only differences are geometric: periodicity is enforced
through lattice vectors and the exact cell area ``|Q|`` is used for averaging —
both inferred from the mesh bounding box and forwarded to the generator.

One call to :func:`fe2_rom.mm.generate_training_data` builds the enrichment
modes φ (N = 3 degenerate buckling modes for the hexagon), derives the
``(F̄, v, g)`` sampling bounds, and runs a multiprocessing pool of full-order
micromorphic solves into ``output_gen/``.

Run as a single process (the pool provides the parallelism):
    python generate_training_data.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.mm import generate_training_data

# --- Configuration ----------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "hexagonal_rve.msh"))
OUTPUT_DIR = os.path.join(HERE, "output_gen")

MAX_STRAIN = 0.10           # intended max per-component deformation
AMPLITUDE_FACTOR = 1.0      # κ: generosity of the v/g bracket
N_SAMPLES = 32
N_WORKERS = 8

E_MICRO, NU_MICRO = 3000.0, 0.30
MU = E_MICRO / (2.0 * (1.0 + NU_MICRO))
LAM = E_MICRO * NU_MICRO / ((1.0 + NU_MICRO) * (1.0 - 2.0 * NU_MICRO))


def hexagon_lattice_from_mesh(path, comm):
    """Infer the pointy-top hexagon ``(lattice_vectors, |Q|)`` from the mesh
    bounding box: a1 = (2ℓ, 0), a2 = (ℓ, √3 ℓ), apothem ℓ; area |Q| = 2√3 ℓ²."""
    ell = None
    if comm.rank == 0:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.open(path)
            xy = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)[:, :2]
        finally:
            gmsh.finalize()
        ell = 0.5 * (xy[:, 0].max() - xy[:, 0].min())
    ell = comm.bcast(ell, root=0)
    s3 = np.sqrt(3.0)
    lattice = np.array([[2.0 * ell, 0.0], [ell, s3 * ell]])
    return lattice, 2.0 * s3 * ell ** 2


def main():
    setup_logging(MPI.COMM_WORLD, level=logging.INFO)
    lattice, rve_volume = hexagon_lattice_from_mesh(RVE_MESH, MPI.COMM_WORLD)
    td = generate_training_data(
        RVE_MESH, MPI.COMM_WORLD, gdim=2, material=NeoHookean(mu=MU, lmbda=LAM),
        max_strain=MAX_STRAIN, lattice_vectors=lattice, rve_volume=rve_volume,
        degree=2, n_samples=N_SAMPLES, amplitude_factor=AMPLITUDE_FACTOR,
        n_workers=N_WORKERS, output_dir=OUTPUT_DIR,
        # φ extraction (only runs if modes/ is absent): equal-biaxial LBA.
        extract_kwargs={"strategy": "lba", "max_strain": MAX_STRAIN},
    )
    print(f"\nTraining data ready in {td.output_dir}")
    print(f"  N modes        : {td.n_modes}")
    print(f"  converged      : {td.n_converged}/{td.n_samples}")
    print(f"  v_max / g_max  : {td.bounds['v_max']} / {td.bounds['g_max']}")
    print(f"  pool           : {td.pool_dir}")
    print(f"  modes          : {td.modes_dir}")


if __name__ == "__main__":
    main()
