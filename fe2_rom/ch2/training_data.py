"""Second-order (CH2) ROM training-data generation.

The second-order analogue of :mod:`fe2_rom.ch1.training_data`: a sample is a
pair ``(F̄, Ḡ)`` (the macroscopic deformation gradient and its gradient). One
call to :func:`generate_training_data` turns the RVE mesh + intended maximum
deformation/gradient into a pool of full-order snapshots (``u_fluc`` + ``P``)
for ``build_rom.py``. The displacement-fluctuation and stress snapshots feed
the POD; the weighted higher-order stress ``𝒴 = ½(Pᵀ⊗x + x⊗P)`` needed for the
second-order ECM block is reconstructed from the ``P`` snapshots in
``build_rom.py`` (it is a closed-form function of ``P`` and the coordinates).

Run as a single process (the multiprocessing pool provides the parallelism)::

    python my_generate_script.py        # NOT mpirun -n N
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from scipy.stats import qmc

from fe2_rom.hyperelastic_solver import setup_logging
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.training_data import (
    TrainingData,
    _fbar_bounds,
    _merge_and_summarize,
    _representative_strain,
)
from fe2_rom.ch2.microsolver import MicroSolver

logger = logging.getLogger("fe2_rom.ch2.training_data")

_DEFAULT_NEWTON = {
    "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 50,
    "div_rel_tol": 10.0, "switch_to_minres": True,
}
_DEFAULT_TIMESTEP = {
    "t_end": 1.0, "dt_init": 1e-1, "dt_min": 1e-5,
    "dt_max": 1e-1, "good_newton_steps": 5,
}
_DEFAULT_SAVE_FIELDS = ("u_fluc", "P")


@dataclass
class _WorkerConfig:
    mesh_path: str
    gdim: int
    degree: int
    material: object
    lattice_vectors: object
    rve_volume: object
    check_stability: bool
    stability_options: dict
    perturb_post_buckling: bool
    pert_amplitude_init: float
    save_fields: tuple
    newton_options: dict
    timestepper_options: dict
    worker_root: str


_CFG: "_WorkerConfig | None" = None


def _init_worker(cfg: _WorkerConfig) -> None:
    global _CFG
    _CFG = cfg
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    setup_logging(MPI.COMM_SELF, level=logging.ERROR)


def _worker_task(task):
    idx, Fbar, G = task
    cfg = _CFG
    work_dir = os.path.join(cfg.worker_root, f"sample_{idx:04d}")
    os.makedirs(work_dir, exist_ok=True)
    try:
        solver = MicroSolver(
            mesh_path=cfg.mesh_path, comm=MPI.COMM_SELF, gdim=cfg.gdim,
            material=cfg.material, degree=cfg.degree,
            output_dir=work_dir, check_stability=cfg.check_stability,
            stability_options=dict(cfg.stability_options),
            perturb_post_buckling=cfg.perturb_post_buckling,
            average_quantities=[],
            visualize_fields=["u_fluc", "P", "u_total"],
            save_snapshots=list(cfg.save_fields),
            newton_options=dict(cfg.newton_options),
            timestepper_options=dict(cfg.timestepper_options),
            lattice_vectors=cfg.lattice_vectors,
            rve_volume=cfg.rve_volume,
        )
        solver(Fbar, G, pert_amplitude_init=cfg.pert_amplitude_init)
        return (idx, True, "")
    except RVEConvergenceError as e:
        return (idx, False, f"RVEConvergenceError: {e}")
    except Exception as e:  # noqa: BLE001
        return (idx, False, f"{type(e).__name__}: {e}")


def _g_bounds(max_grad, gdim):
    """Per-component symmetric bound ``g_amp`` for the gdim³ components of Ḡ.

    ``max_grad`` is a scalar amplitude (every ``Ḡ_iJK ∈ [−max_grad, max_grad]``,
    units 1/length).
    """
    return float(max_grad)


def generate_training_data(
    mesh_path: str, comm, gdim: int, material,
    max_strain: "float | dict", max_grad: float, *,
    lattice_vectors=None, degree: int = 1, n_samples: int = 64,
    sampler: str = "lhs", seed: int = 12345, n_workers: "int | None" = None,
    check_stability: bool = False, stability_options: "dict | None" = None,
    perturb_post_buckling: bool = False, pert_amplitude_init: float = 1e-2,
    newton_options=None, timestepper_options=None,
    save_fields=_DEFAULT_SAVE_FIELDS, rve_volume: "float | None" = None,
    output_dir: "str | None" = None,
) -> TrainingData:
    """Generate second-order (CH2) ROM training data.

    Samples ``(F̄, Ḡ)`` jointly by Latin hypercube over the ``F̄`` box (from
    ``max_strain``, same convention as ``ch1.generate_training_data``) and the
    ``Ḡ`` box ``[−max_grad, max_grad]`` (symmetric in Ḡ's last two indices), then
    drives the full-order ``ch2.MicroSolver`` from ``(I, 0)`` to each ``(F̄, Ḡ)``
    on a ``t: 0→1`` ramp, saving ``u_fluc``/``P`` snapshots. Returns the usual
    ``TrainingData`` summary; ``targets`` additionally carries the ``G`` array.
    """
    if comm.size > 1:
        logger.warning("generate_training_data is single-process (use plain python).")
    output_dir = output_dir or os.path.abspath("ch2_training_output")
    pool_dir = os.path.join(output_dir, "snapshots_pool")
    worker_root = os.path.join(output_dir, "workers")
    os.makedirs(output_dir, exist_ok=True)

    F_lo, F_hi = _fbar_bounds(max_strain, gdim)
    g_amp = _g_bounds(max_grad, gdim)
    strain_amp = _representative_strain(F_lo, F_hi, gdim)
    bounds = {"max_strain": max_strain, "strain_amp": strain_amp,
              "F_lo": F_lo, "F_hi": F_hi, "g_amp": g_amp}
    logger.info("Bounds: strain(amp)=%.4g  grad(amp)=%.4g", strain_amp, g_amp)

    # Joint LHS over the gdim² F̄ components and the gdim·gdim·(gdim+1)/2
    # independent Ḡ components (last two indices symmetric).
    g_pairs = [(i, J, K) for i in range(gdim)
               for J in range(gdim) for K in range(J, gdim)]
    dim = gdim * gdim + len(g_pairs)
    if sampler == "lhs":
        unit = qmc.LatinHypercube(d=dim, seed=seed).random(n_samples)
    elif sampler == "sobol":
        unit = qmc.Sobol(d=dim, seed=seed).random(n_samples)
    else:
        unit = np.random.default_rng(seed).random((n_samples, dim))

    samples_F, samples_G = [], []
    for row in unit:
        Fbar = (F_lo.ravel() + row[:gdim * gdim]
                * (F_hi.ravel() - F_lo.ravel())).reshape(gdim, gdim)
        G = np.zeros((gdim, gdim, gdim))
        for k, (i, J, K) in enumerate(g_pairs):
            val = (2.0 * row[gdim * gdim + k] - 1.0) * g_amp
            G[i, J, K] = G[i, K, J] = val
        samples_F.append(Fbar)
        samples_G.append(G)

    if worker_root and os.path.isdir(worker_root):
        shutil.rmtree(worker_root)
    os.makedirs(worker_root, exist_ok=True)
    cfg = _WorkerConfig(
        mesh_path=mesh_path, gdim=gdim, degree=degree, material=material,
        lattice_vectors=lattice_vectors, rve_volume=rve_volume,
        check_stability=check_stability,
        stability_options=stability_options or {"nev": 5, "neg_tol": -1e-8},
        perturb_post_buckling=perturb_post_buckling,
        pert_amplitude_init=pert_amplitude_init,
        save_fields=tuple(save_fields),
        newton_options=newton_options or _DEFAULT_NEWTON,
        timestepper_options=timestepper_options or _DEFAULT_TIMESTEP,
        worker_root=worker_root,
    )
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    tasks = [(i, samples_F[i], samples_G[i]) for i in range(n_samples)]
    logger.info("Running %d (F̄,Ḡ) sample(s) on %d worker(s) …", n_samples, n_workers)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        results = pool.map(_worker_task, tasks, chunksize=1)

    ok_flags = np.zeros(n_samples, dtype=bool)
    for idx, ok, msg in results:
        ok_flags[idx] = ok
        if not ok:
            logger.warning("  sample %04d failed: %s", idx, msg)
    n_converged = int(ok_flags.sum())
    logger.info("Converged %d/%d sample(s)", n_converged, n_samples)

    n_copied, realized = _merge_and_summarize(worker_root, pool_dir, samples_F, gdim)
    targets = {"Fbar": np.stack(samples_F), "G": np.stack(samples_G), "ok": ok_flags}
    np.savez(os.path.join(output_dir, "sample_inputs.npz"), **targets)
    logger.info("Pooled %d snapshot file(s) → %s", n_copied, pool_dir)

    return TrainingData(
        output_dir=output_dir, pool_dir=pool_dir,
        n_samples=n_samples, n_converged=n_converged,
        bounds=bounds, realized_ranges=realized, targets=targets,
    )
