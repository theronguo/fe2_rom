"""Stage 2: micromorphic ROM training-data generation for the square RVE.

One call to :func:`fe2_rom.mm.generate_training_data`:
  1. builds the enrichment modes φ via ``extract_buckling_modes`` (if not cached
     under ``output_gen/modes/``),
  2. derives the ``(F̄, v, g)`` sampling bounds from ``MAX_STRAIN`` and the RVE
     size (generous bracket; convergence filters the rest),
  3. runs a multiprocessing pool of full-order micromorphic solves, and
  4. writes a snapshot pool ready for ``build_rom.py``.

Run as a single process (the pool provides the parallelism):
    python generate_training_data.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging

from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.mm import generate_training_data

# --- Configuration ----------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "rve.msh"))
OUTPUT_DIR = os.path.join(HERE, "output_gen")

MAX_STRAIN = 0.10           # intended max per-component deformation
AMPLITUDE_FACTOR = 2.0      # κ: generosity of the v/g bracket
N_SAMPLES = 32
N_WORKERS = 8

E_MICRO, NU_MICRO = 3000.0, 0.30
MU = E_MICRO / (2.0 * (1.0 + NU_MICRO))
LAM = E_MICRO * NU_MICRO / ((1.0 + NU_MICRO) * (1.0 - 2.0 * NU_MICRO))


def main():
    setup_logging(MPI.COMM_WORLD, level=logging.INFO)
    td = generate_training_data(
        RVE_MESH, MPI.COMM_WORLD, gdim=2, material=NeoHookean(mu=MU, lmbda=LAM),
        max_strain=MAX_STRAIN, lattice_vectors=None, degree=2,
        n_samples=N_SAMPLES, amplitude_factor=AMPLITUDE_FACTOR,
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
