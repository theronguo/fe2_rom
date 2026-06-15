"""First-order (CH1) ROM training-data generation.

The classical-homogenization analogue of :mod:`fe2_rom.mm.training_data`: there
are no enrichment modes φ and no ``(v, g)`` macro variables, so a sample is just
a target ``F̄``. One call to :func:`generate_training_data` turns a mesh + an
intended maximum deformation into a pool of full-order snapshots
(``u_fluc`` + ``P``) ready for ``build_rom.py``:

  1. **Sampling bounds.** Derived from ``max_strain`` (scalar symmetric box, or a
     per-component dict of absolute ranges — same convention as the micromorphic
     generator).
  2. **Space-filling samples.** ``n_samples`` targets ``F̄`` from a Latin
     hypercube over the bound box (optionally the symmetric stretch ``U`` only,
     to train a ROM for the objectivity reduction).
  3. **Parallel FOM solves.** A ``multiprocessing`` pool of independent
     ``MPI.COMM_SELF`` workers each drives the full-order
     :class:`fe2_rom.ch1.microsolver.MicroSolver` from ``I`` to its sampled
     ``F̄`` (a ``t: 0→1`` ramp), saving the snapshot fields along the way. Any
     sample whose endpoint overshoots the reachable set simply fails to converge
     and is discarded, so the converged-snapshot cloud traces the physically
     reachable ``F̄`` region (reported back).
  4. **Merge + report.** Worker snapshots are merged into ``snapshots_pool/``
     (the layout ``build_rom.py`` globs).

Run as a single process (the pool provides the parallelism)::

    python my_generate_script.py        # NOT mpirun -n N
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob

import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import setup_logging
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.ch1.microsolver import MicroSolver

logger = logging.getLogger("fe2_rom.ch1.training_data")

_DEFAULT_NEWTON = {
    "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 50,
    "div_rel_tol": 10.0, "switch_to_minres": True,
}
_DEFAULT_TIMESTEP = {
    "t_end": 1.0, "dt_init": 1e-1, "dt_min": 1e-5,
    "dt_max": 1e-1, "good_newton_steps": 5,
}
# Snapshot fields consumed by the CH1 build_rom.py (u_fluc + P). Extend via
# ``save_fields``.
_DEFAULT_SAVE_FIELDS = ("u_fluc", "P")


# ---------------------------------------------------------------------------
# Public result
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Summary of a :func:`generate_training_data` run."""
    output_dir: str
    pool_dir: str          # snapshots_pool/  (input to build_rom.py)
    n_samples: int
    n_converged: int
    bounds: dict           # {"max_strain", "strain_amp", "F_lo", "F_hi"}
    realized_ranges: dict  # {"Fbar": (lo, hi)} of converged snapshots
    targets: dict          # {"Fbar", "ok"} arrays, one row per sample


# ---------------------------------------------------------------------------
# Worker (multiprocessing, one MPI.COMM_SELF solver per process)
# ---------------------------------------------------------------------------

@dataclass
class _WorkerConfig:
    mesh_path: str
    gdim: int
    degree: int
    material: object
    lattice_vectors: object
    rve_volume: object
    corner_periodic: bool
    check_stability: bool
    perturb_post_buckling: bool
    pert_amplitude_init: float
    save_fields: tuple
    newton_options: dict
    timestepper_options: dict
    worker_root: str


_CFG: "_WorkerConfig | None" = None


def _init_worker(cfg: _WorkerConfig) -> None:
    """Pool initializer: stash the (picklable) config and cap BLAS threads so
    the pool doesn't oversubscribe."""
    global _CFG
    _CFG = cfg
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    setup_logging(MPI.COMM_SELF, level=logging.ERROR)


def _worker_task(task):
    """Run one F̄ sample to t=1 (or until it diverges), saving snapshots into the
    worker's private ``output_dir/snapshots/``."""
    idx, Fbar = task
    cfg = _CFG
    work_dir = os.path.join(cfg.worker_root, f"sample_{idx:04d}")
    os.makedirs(work_dir, exist_ok=True)
    try:
        solver = MicroSolver(
            mesh_path=cfg.mesh_path, comm=MPI.COMM_SELF, gdim=cfg.gdim,
            material=cfg.material, degree=cfg.degree,
            output_dir=work_dir, check_stability=cfg.check_stability,
            perturb_post_buckling=cfg.perturb_post_buckling,
            # Training-data solves are pure forward problems: an empty
            # average_quantities list means no effective quantities / tangents
            # and no macro-sensitivity (adjoint) solves are done.
            average_quantities=[],
            # "P" must be in visualize_fields for the P snapshot path to allocate
            # self.P_func.
            visualize_fields=["u_fluc", "P", "u_total"],
            save_snapshots=list(cfg.save_fields),
            newton_options=dict(cfg.newton_options),
            timestepper_options=dict(cfg.timestepper_options),
            corner_periodic=cfg.corner_periodic,
            lattice_vectors=cfg.lattice_vectors,
            rve_volume=cfg.rve_volume,
        )
        solver(Fbar, pert_amplitude_init=cfg.pert_amplitude_init)
        return (idx, True, "")
    except RVEConvergenceError as e:
        return (idx, False, f"RVEConvergenceError: {e}")
    except Exception as e:  # noqa: BLE001 — isolate worker failures
        return (idx, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# F̄ sampling box
# ---------------------------------------------------------------------------

_AXIS = {"x": 0, "y": 1, "z": 2}


def _parse_fbar_key(key, gdim):
    """Parse an F̄-component key like ``"F_xx"`` / ``"Fxy"`` / ``"xy"`` into a
    ``(i, j)`` index pair (x→0, y→1, z→2)."""
    s = str(key)
    if s.startswith("F_"):
        s = s[2:]
    elif s.startswith("F"):
        s = s[1:]
    if len(s) != 2 or s[0].lower() not in _AXIS or s[1].lower() not in _AXIS:
        raise ValueError(f"F̄ component key {key!r} must be like 'F_xx' "
                         f"(two axis letters from x/y/z)")
    i, j = _AXIS[s[0].lower()], _AXIS[s[1].lower()]
    if i >= gdim or j >= gdim:
        raise ValueError(f"F̄ component key {key!r} out of range for gdim={gdim}")
    return i, j


def _fbar_bounds(max_strain, gdim):
    """Per-component ``(F_lo, F_hi)`` ``(gdim, gdim)`` sampling bounds for F̄.

    * scalar ``ε`` → ``F̄ = I + δ`` with every ``δ_ij ∈ [−ε, ε]`` (symmetric box).
    * dict → per-component *absolute* ranges, e.g.
      ``{"F_xx": [0.85, 1.05], "F_xy": [-0.1, 0.1]}``; components not listed are
      held fixed at the identity (diagonal 1, off-diagonal 0)."""
    eye = np.eye(gdim)
    if isinstance(max_strain, dict):
        F_lo, F_hi = eye.copy(), eye.copy()
        for key, rng in max_strain.items():
            i, j = _parse_fbar_key(key, gdim)
            lo, hi = float(rng[0]), float(rng[1])
            F_lo[i, j], F_hi[i, j] = min(lo, hi), max(lo, hi)
        return F_lo, F_hi
    eps = float(max_strain)
    return eye - eps, eye + eps


def _representative_strain(F_lo, F_hi, gdim):
    """Largest deviation of the F̄ box from the identity."""
    eye = np.eye(gdim)
    return float(np.maximum(np.abs(F_lo - eye), np.abs(F_hi - eye)).max())


def _draw_samples(n_samples, gdim, F_lo, F_hi, seed, sampler, symmetric=False):
    """``n_samples`` targets ``F̄`` space-filling over the bound box: each ``F̄``
    component independently in ``[F_lo, F_hi]``.

    ``symmetric=True`` samples only the **symmetric stretch ``U``**: the LHS fills
    the ``gdim*(gdim+1)/2`` independent components (3 in 2D, 6 in 3D) from the
    *upper-triangle* ``[F_lo, F_hi]`` ranges, and the lower triangle is mirrored
    so every ``F̄`` is symmetric. This is the training counterpart of the
    objectivity reduction (drive RVEs with ``U`` online): the snapshots are then
    rotation-free (``R = I``) and the sampling space drops from 9 to 6 dims."""
    pairs = [(p, q) for p in range(gdim) for q in range(p, gdim)]
    dim = len(pairs) if symmetric else gdim * gdim
    F_lo = np.asarray(F_lo)
    F_hi = np.asarray(F_hi)

    if sampler in ("lhs", "sobol"):
        from scipy.stats import qmc
        engine = (qmc.LatinHypercube(d=dim, seed=seed) if sampler == "lhs"
                  else qmc.Sobol(d=dim, seed=seed))
        unit = engine.random(n_samples)
    elif sampler == "uniform":
        unit = np.random.default_rng(seed).random((n_samples, dim))
    else:
        raise ValueError(f"sampler must be 'lhs', 'sobol' or 'uniform', got {sampler!r}")

    samples = []
    for row in unit:
        if symmetric:
            Fbar = np.zeros((gdim, gdim))
            for k, (p, q) in enumerate(pairs):
                val = F_lo[p, q] + row[k] * (F_hi[p, q] - F_lo[p, q])
                Fbar[p, q] = Fbar[q, p] = val
        else:
            Fbar = (F_lo.ravel() + row * (F_hi.ravel() - F_lo.ravel())).reshape(gdim, gdim)
        samples.append(Fbar)
    return samples


_T_RE = re.compile(r"^(?P<field>.+?)_(?P<t>[0-9]+\.[0-9]+)\.npy$")


def _merge_and_summarize(worker_root, pool_dir, samples, gdim):
    """Flatten per-worker snapshots into ``pool_dir`` and compute the realized
    ``F̄`` range from the converged snapshots.

    Each saved snapshot at load parameter ``t`` corresponds to the ramped state
    ``I + t(F̄−I)``, so the realized range follows from the sample targets and the
    set of ``t`` actually reached (= the steps that converged)."""
    if os.path.isdir(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)

    eye = np.eye(gdim)
    n_copied = 0
    realized_F = []

    for sample_dir in sorted(glob(os.path.join(worker_root, "sample_*"))):
        try:
            idx = int(os.path.basename(sample_dir).split("_")[-1])
        except ValueError:
            continue
        snap_dir = os.path.join(sample_dir, "snapshots")
        if not os.path.isdir(snap_dir):
            continue
        Fbar_t = samples[idx]
        for f in sorted(glob(os.path.join(snap_dir, "*.npy"))):
            name = os.path.basename(f)
            if name.endswith("_dof_coords.npy"):
                tgt = os.path.join(pool_dir, name)        # shared, copy once
                if not os.path.exists(tgt):
                    shutil.copy(f, tgt)
                continue
            m = _T_RE.match(name)
            if not m:
                continue
            field_name, t_str = m.group("field"), m.group("t")
            shutil.copy(f, os.path.join(pool_dir, f"{field_name}_s{idx:04d}_{t_str}.npy"))
            n_copied += 1
            if field_name == "u_fluc":          # one realized state per u snapshot
                t = float(t_str)
                realized_F.append(eye + t * (Fbar_t - eye))

    if realized_F:
        a = np.stack(realized_F)
        realized = {"Fbar": (a.min(axis=0), a.max(axis=0))}
    else:
        nan = np.full((gdim, gdim), np.nan)
        realized = {"Fbar": (nan, nan.copy())}
    return n_copied, realized


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_training_data(
    mesh_path: str, comm, gdim: int, material, max_strain: "float | dict", *,
    lattice_vectors: "np.ndarray | None" = None,
    degree: int = 1,
    n_samples: int = 64,
    sampler: str = "lhs",
    sample_symmetric_stretch: bool = False,
    seed: int = 12345,
    n_workers: "int | None" = None,
    corner_periodic: bool = False,
    check_stability: bool = False,
    perturb_post_buckling: bool = False,
    pert_amplitude_init: float = 1e-2,
    newton_options: "dict | None" = None,
    timestepper_options: "dict | None" = None,
    save_fields: "tuple | list" = _DEFAULT_SAVE_FIELDS,
    rve_volume: "float | None" = None,
    output_dir: "str | None" = None,
) -> TrainingData:
    """Generate first-order (CH1) ROM training data for an RVE.

    Parameters
    ----------
    mesh_path, comm, gdim, material : the RVE (as for ``ch1.MicroSolver``). The
        driver runs as a single process and uses ``MPI.COMM_SELF`` internally;
        ``comm`` is only used for logging, so run this with plain ``python`` (the
        pool provides the parallelism), not ``mpirun``.
    max_strain : the intended maximum deformation. Either a **scalar** ``ε`` →
        symmetric box ``F̄ = I + δ`` with every ``δ_ij ∈ [−ε, ε]``, or a **dict**
        of per-component *absolute* ranges, e.g.
        ``{"F_xx": [0.85, 1.05], "F_xy": [-0.1, 0.1]}`` (keys ``"F_<ab>"`` with
        axes ``x``/``y``/``z``); components not listed are held at the identity.
    lattice_vectors : ``None`` for an axis-aligned box, ``(gdim, gdim)`` for a
        periodic polygon (forwarded to the solver).
    degree : RVE displacement (FE) order. Keep consistent with ``build_rom.py``
        and the online reduced solver.
    n_samples : number of ``F̄`` samples.
    sampler : ``"lhs"`` (default), ``"sobol"`` or ``"uniform"``.
    sample_symmetric_stretch : if ``True``, sample only the symmetric stretch
        ``U`` (the ``gdim*(gdim+1)/2`` independent components, lower triangle
        mirrored) instead of the full ``F̄``. Use this to train a ROM for the
        objectivity reduction (``objective_reduction=True`` on the online
        solver): snapshots are rotation-free and the space drops from 9 to 6 dims.
    corner_periodic : RVE gauge for the sample solves — must match the gauge the
        reduced solver / FOM macro driver use. ``False`` (default) pins the
        corner dofs to zero (classical CH1); ``True`` uses the ``⟨w⟩=0`` integral
        constraint. Both apply periodic face ties.
    seed, n_workers : RNG seed and pool size (default ``cpu_count − 1``).
    check_stability : monitor RVE stability during the sample solves. Default
        ``False`` — we want the equilibrium fluctuation at the *prescribed* ``F̄``.
    perturb_post_buckling : when an unstable equilibrium is detected (requires
        ``check_stability=True``), perturb along the lowest eigenmode and re-solve
        so the ramp follows the *buckled* branch instead of stalling at the
        bifurcation. Set ``True`` to put post-buckling states into the snapshot
        pool — otherwise (``False``, default) the solver halves ``dt`` to approach
        the bifurcation and the sample is discarded once it can go no further, so
        the ROM only ever sees the pre-buckling branch.
    pert_amplitude_init : initial eigenmode-kick amplitude used when
        ``perturb_post_buckling`` traverses a bifurcation (default ``1e-2``). The
        usable window is narrow and RVE-dependent: too large overshoots Newton's
        basin and the kicked solve diverges; too small fails to escape the
        near-singular bifurcation point. For thin-strut RVEs that buckle early,
        ``~1e-3`` traverses simple (isolated-mode) bifurcations where the default
        ``1e-2`` does not. Degenerate (e.g. equibiaxial, square-symmetric)
        bifurcations cannot be traversed by a single-eigenvector kick at any
        amplitude. Ignored when ``perturb_post_buckling=False``.
    newton_options, timestepper_options : inner-solve options (sensible defaults).
    save_fields : snapshot fields to save (default ``("u_fluc", "P")``).
    rve_volume : exact cell volume ``|Q|`` (required for porous polygon cells).
    output_dir : root for ``snapshots_pool/``, ``workers/`` and the summary
        ``.npz`` files (default ``./ch1_training_output``).

    Returns
    -------
    TrainingData with the bounds used, the realized ``F̄`` range of the converged
    snapshots, and the path consumed by ``build_rom.py``.
    """
    if comm.size > 1:
        logger.warning("generate_training_data is a single-process driver "
                       "(multiprocessing pool); comm.size=%d — run with plain "
                       "`python`, not mpirun.", comm.size)

    output_dir = output_dir or os.path.abspath("ch1_training_output")
    pool_dir = os.path.join(output_dir, "snapshots_pool")
    worker_root = os.path.join(output_dir, "workers")
    os.makedirs(output_dir, exist_ok=True)

    # F̄ sampling box.
    F_lo, F_hi = _fbar_bounds(max_strain, gdim)
    strain_amp = _representative_strain(F_lo, F_hi, gdim)
    bounds = {"max_strain": max_strain, "strain_amp": strain_amp,
              "F_lo": F_lo, "F_hi": F_hi}
    with np.printoptions(precision=4, suppress=True):
        logger.info("Bounds: strain(amp)=%.4g", strain_amp)
        logger.info("        F̄ box lo=\n%s\n        F̄ box hi=\n%s", F_lo, F_hi)

    # Space-filling samples.
    if sample_symmetric_stretch:
        logger.info("Sampling the SYMMETRIC stretch U only (objectivity "
                    "reduction): %d-dim F̄ box, rotation-free snapshots.",
                    gdim * (gdim + 1) // 2)
    samples = _draw_samples(n_samples, gdim, F_lo, F_hi, seed, sampler,
                            symmetric=sample_symmetric_stretch)

    # Parallel FOM solves (multiprocessing pool, COMM_SELF per worker).
    if worker_root and os.path.isdir(worker_root):
        shutil.rmtree(worker_root)
    os.makedirs(worker_root, exist_ok=True)
    cfg = _WorkerConfig(
        mesh_path=mesh_path, gdim=gdim, degree=degree, material=material,
        lattice_vectors=lattice_vectors, rve_volume=rve_volume,
        corner_periodic=corner_periodic, check_stability=check_stability,
        perturb_post_buckling=perturb_post_buckling,
        pert_amplitude_init=pert_amplitude_init,
        save_fields=tuple(save_fields),
        newton_options=newton_options or _DEFAULT_NEWTON,
        timestepper_options=timestepper_options or _DEFAULT_TIMESTEP,
        worker_root=worker_root,
    )
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    tasks = [(i, Fbar) for i, Fbar in enumerate(samples)]
    logger.info("Running %d sample(s) on %d worker(s) …", n_samples, n_workers)
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

    # Merge + realized range + side-tables.
    n_copied, realized = _merge_and_summarize(worker_root, pool_dir, samples, gdim)
    targets = {"Fbar": np.stack(samples), "ok": ok_flags}
    np.savez(os.path.join(output_dir, "sample_inputs.npz"), **targets)
    np.savez(os.path.join(output_dir, "training_summary.npz"),
             strain_amp=strain_amp, F_lo=F_lo, F_hi=F_hi,
             realized_Fbar_lo=realized["Fbar"][0], realized_Fbar_hi=realized["Fbar"][1])

    logger.info("Pooled %d snapshot file(s) → %s", n_copied, pool_dir)
    with np.printoptions(precision=4, suppress=True):
        logger.info("Realized F̄ range (converged): [%s,\n %s]",
                    realized["Fbar"][0], realized["Fbar"][1])

    return TrainingData(
        output_dir=output_dir, pool_dir=pool_dir,
        n_samples=n_samples, n_converged=n_converged,
        bounds=bounds, realized_ranges=realized, targets=targets,
    )
