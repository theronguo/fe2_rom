"""Stage 2: second-order (CH2) ROM training-data generation for the square RVE.

The CH2 analogue of ``examples/mm/example_1/generate_training_data.py``. One call
to :func:`fe2_rom.ch2.generate_training_data`:

  1. derives the sampling box from ``MAX_STRAIN`` (the ``F̄`` box ``I ± ε``) and
     ``MAX_GRAD`` (the ``Ḡ`` box ``[−max_grad, max_grad]``, symmetric in Ḡ's last
     two indices),
  2. draws ``N_SAMPLES`` Latin-hypercube pairs ``(F̄, Ḡ)``,
  3. runs a multiprocessing pool of full-order second-order RVE solves (each on
     ``MPI.COMM_SELF``, ramping ``(I, 0) → (F̄, Ḡ)``), and
  4. writes a ``u_fluc`` + ``P`` snapshot pool ready for ``build_rom.py``.

The double-/couple-stress density ``𝒴 = ½(X_K P_iJ + X_J P_iK)`` is reconstructed
from the ``P`` snapshots in ``build_rom.py`` (a closed-form function of ``P`` and
the coordinates), so only ``u_fluc`` and ``P`` are saved here.

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
from fe2_rom.ch2 import generate_training_data

# --- Configuration ----------------------------------------------------------
HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "rve.msh"))
OUTPUT_DIR = os.path.join(HERE, "output_gen")

MAX_STRAIN = 0.10           # F̄ box: every δ_ij ∈ [−0.10, 0.10]
MAX_GRAD = 0.05             # Ḡ box: every Ḡ_iJK ∈ [−0.05, 0.05]  (1/length)
N_SAMPLES = 64              # (F̄, Ḡ) is a 10-D input space — sample it densely
N_WORKERS = 12

E_MICRO, NU_MICRO = 3000.0, 0.30
MU = E_MICRO / (2.0 * (1.0 + NU_MICRO))
LAM = E_MICRO * NU_MICRO / ((1.0 + NU_MICRO) * (1.0 - 2.0 * NU_MICRO))


def main():
    setup_logging(MPI.COMM_WORLD, level=logging.INFO)
    td = generate_training_data(
        RVE_MESH, MPI.COMM_WORLD, gdim=2, material=NeoHookean(mu=MU, lmbda=LAM),
        max_strain=MAX_STRAIN, max_grad=MAX_GRAD,
        lattice_vectors=None, degree=2,
        n_samples=N_SAMPLES, n_workers=N_WORKERS, output_dir=OUTPUT_DIR,
        # Monitor stability and traverse buckling for every sample: this RVE
        # buckles from ~2.8% (near-)equibiaxial compression, so without this the
        # snapshots past the bifurcation would sit on the unphysical unbuckled
        # branch. The eigenmode kick must start small — the solver default
        # (1e-2) overshoots Newton's basin for this thin-ligament RVE (the retry
        # loop then only doubles it); 1e-3 lands on the buckled branch and
        # traverses cleanly to t=1 (verified for both asymmetric and equibiaxial
        # compression).
        check_stability=True,
        perturb_post_buckling=True,
        pert_amplitude_init=1e-3,
    )
    print(f"\nTraining data ready in {td.output_dir}")
    print(f"  converged      : {td.n_converged}/{td.n_samples}")
    print(f"  strain / grad  : {td.bounds['strain_amp']} / {td.bounds['g_amp']}")
    print(f"  pool           : {td.pool_dir}")


if __name__ == "__main__":
    main()
