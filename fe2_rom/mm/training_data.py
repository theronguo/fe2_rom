"""Stage 2: training-data generation for the micromorphic ROM.

One call — :func:`generate_training_data` — turns a mesh + an intended maximum
deformation into a pool of full-order micromorphic snapshots ready for
``build_rom.py``:

  1. **Enrichment modes φ.** Built via :func:`fe2_rom.mm.extract_buckling_modes`
     if not already on disk (cached under ``modes_dir/phi/``).
  2. **Sampling bounds.** Derived automatically from the intended ``max_strain``
     and the RVE size ``L_RVE`` (see below) — no hand-tuned ranges.
  3. **Space-filling samples.** ``n_samples`` triples ``(F̄, v, g)`` from a Latin
     hypercube over the bound box.
  4. **Parallel FOM solves.** A ``multiprocessing`` pool of independent
     ``MPI.COMM_SELF`` workers each drives the micromorphic ``MicroSolver`` from
     ``I`` to its sampled ``(F̄, v, g)`` (a ``t: 0→1`` ramp), saving the snapshot
     fields along the way.
  5. **Merge + report.** Worker snapshots are merged into ``snapshots_pool/``
     (the layout ``build_rom.py`` globs), and the *realized* ``(F̄, v, g)`` ranges
     of the converged snapshots are reported.

Bounds (generous bracket — convergence is the real filter)
----------------------------------------------------------
``v·φ`` is the pattern contribution to the displacement, so a physically
meaningful amplitude bound asks the *mean pattern displacement* ``|v|·⟨‖φ‖⟩`` to
reach ``≈ a few × the cell shortening`` at max load. Solving for ``v`` makes the
bound **invariant to the φ normalisation** (buckling φ are H¹-orthonormal, not
mean-normalised), per mode ``i``::

    v_max[i] = amplitude_factor · max_strain · L_RVE / ⟨‖φ_i‖⟩_Q
    g_max[i] = v_max[i] / L_RVE

Because each sample is reached by a ``t``-ramp from the reference state, the
ramps fill the box interior automatically, and any sample whose endpoint
overshoots the kinematically reachable set simply fails to converge and is
discarded — so ``v_max`` only has to be a (deliberate) *over-estimate*. The
converged-snapshot cloud then traces the physically reachable ``(F̄, v, g)``
region, which is reported back. This needs no post-buckling traverse, so it
works fine with LBA-only enrichment modes.

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
from fe2_rom.mm.microsolver import MicroSolver
from fe2_rom.mm.enrichment_modes import extract_buckling_modes

logger = logging.getLogger("fe2_rom.mm.training_data")

_DEFAULT_NEWTON = {
    "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 50,
    "div_rel_tol": 10.0, "switch_to_minres": True,
}
_DEFAULT_TIMESTEP = {
    "t_end": 1.0, "dt_init": 1e-1, "dt_min": 1e-5,
    "dt_max": 1e-1, "good_newton_steps": 5,
}
# Snapshot fields consumed by the micromorphic build_rom.py (u_fluc + P; Π/Λ
# densities are reconstructed there from P and φ). Extend via ``save_fields``.
_DEFAULT_SAVE_FIELDS = ("u_fluc", "P")


# ---------------------------------------------------------------------------
# Public result
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Summary of a :func:`generate_training_data` run."""
    output_dir: str
    pool_dir: str          # snapshots_pool/  (input to build_rom.py)
    modes_dir: str         # modes/           (φ live in modes/phi/)
    n_modes: int
    n_samples: int
    n_converged: int
    bounds: dict           # {"max_strain", "L_RVE", "amplitude_factor", "v_max", "g_max"}
    realized_ranges: dict  # {"Fbar": (lo, hi), "v": (lo, hi), "g": (lo, hi)} of converged snapshots
    targets: dict          # {"Fbar", "v", "g", "ok"} arrays, one row per sample


# ---------------------------------------------------------------------------
# Worker (multiprocessing, one MPI.COMM_SELF solver per process)
# ---------------------------------------------------------------------------

@dataclass
class _WorkerConfig:
    mesh_path: str
    gdim: int
    degree: int
    material: object
    n_modes: int
    phi_arrays: list
    lattice_vectors: object
    rve_volume: object
    check_stability: bool
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
    """Run one (F̄, v, g) sample to t=1 (or until it diverges), saving snapshots
    into the worker's private ``output_dir/snapshots/``."""
    idx, Fbar, v, g = task
    cfg = _CFG
    work_dir = os.path.join(cfg.worker_root, f"sample_{idx:04d}")
    os.makedirs(work_dir, exist_ok=True)
    try:
        solver = MicroSolver(
            mesh_path=cfg.mesh_path, comm=MPI.COMM_SELF, gdim=cfg.gdim,
            material=cfg.material, N=cfg.n_modes, degree=cfg.degree,
            output_dir=work_dir, check_stability=cfg.check_stability,
            perturb_post_buckling=False,
            # Training-data solves are pure forward problems: an empty
            # average_quantities list means _collect_averages computes nothing,
            # so no effective quantities / tangents and no macro-sensitivity
            # (adjoint) solves are done; combined with check_stability=False and
            # saving only the displacement + stress fields.
            average_quantities=[],
            # "P" must be in visualize_fields for the P snapshot path to allocate
            # self.P_func; "A" likewise for the rank-4 tangent.
            visualize_fields=[f for f in ("P", "A") if f in cfg.save_fields],
            save_snapshots=list(cfg.save_fields),
            newton_options=dict(cfg.newton_options),
            timestepper_options=dict(cfg.timestepper_options),
            lattice_vectors=cfg.lattice_vectors,
            rve_volume=cfg.rve_volume,
        )
        solver.load_buckling_modes([cfg.phi_arrays[i] for i in range(cfg.n_modes)])
        solver(Fbar, v, g)
        return (idx, True, "")
    except RVEConvergenceError as e:
        return (idx, False, f"RVEConvergenceError: {e}")
    except Exception as e:  # noqa: BLE001 — isolate worker failures
        return (idx, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rve_length_scale(mesh_path: str, gdim: int) -> float:
    """Mean bounding-box extent of the mesh — the characteristic cell size L_RVE."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(mesh_path)
        xyz = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)[:, :gdim]
    finally:
        gmsh.finalize()
    ext = xyz.max(axis=0) - xyz.min(axis=0)
    return float(np.mean(ext))


def _phi_mean_norms(mesh_path: str, gdim: int, degree: int,
                    phi_arrays: list[np.ndarray]) -> np.ndarray:
    """Per-mode mean pointwise norm ``⟨‖φ_i‖⟩_Q = (1/|Q|) ∫ ‖φ_i‖ dX`` — used to
    make the ``v`` bound invariant to the φ normalisation."""
    import ufl
    from dolfinx import fem, io
    from fe2_rom.hyperelastic_solver.logging_utils import silence_c_stdout
    with silence_c_stdout():
        mesh = io.gmsh.read_from_msh(mesh_path, MPI.COMM_SELF, 0, gdim=gdim).mesh
    V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    dx = ufl.Measure("dx", domain=mesh)
    vol = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * dx))
    phi = fem.Function(V)
    norms = []
    for arr in phi_arrays:
        phi.x.array[:] = arr
        phi.x.scatter_forward()
        integ = fem.assemble_scalar(fem.form(ufl.sqrt(ufl.dot(phi, phi)) * dx))
        norms.append(integ / vol)
    return np.array(norms)


def _load_phi_arrays(phi_dir: str) -> list[np.ndarray]:
    files = sorted(
        f for f in glob(os.path.join(phi_dir, "phi_*.npy"))
        if "dof_coords" not in os.path.basename(f)
        and "singular_values" not in os.path.basename(f)
    )
    return [np.load(f) for f in files]


def _ensure_modes(mesh_path, comm, gdim, material, *, degree, lattice_vectors,
                  modes_dir, n_modes, extract_kwargs) -> list[np.ndarray]:
    """Load cached φ from ``modes_dir/phi/`` or build them with
    :func:`extract_buckling_modes`."""
    phi_dir = os.path.join(modes_dir, "phi")
    cached = _load_phi_arrays(phi_dir)
    if cached:
        logger.info("Using %d cached φ mode(s) from %s", len(cached), phi_dir)
    else:
        logger.info("No cached φ found — extracting buckling modes into %s", modes_dir)
        kw = dict(degree=degree, lattice_vectors=lattice_vectors,
                  output_dir=modes_dir)
        kw.update(extract_kwargs or {})
        extract_buckling_modes(mesh_path, comm, gdim, material, **kw)
        cached = _load_phi_arrays(phi_dir)
        if not cached:
            raise RuntimeError(f"extract_buckling_modes wrote no φ files to {phi_dir}")
    if n_modes is not None:
        if n_modes > len(cached):
            raise ValueError(f"requested n_modes={n_modes} but only {len(cached)} "
                             f"φ mode(s) available in {phi_dir}")
        cached = cached[:n_modes]
    return cached


def _draw_samples(n_samples, n_modes, gdim, max_strain, v_max, g_max,
                  seed, sampler):
    """``n_samples`` triples ``(F̄, v, g)`` space-filling over the bound box.
    Each component is symmetric about its reference value: ``F̄ = I + δ`` with
    ``δ_ij ∈ [−max_strain, max_strain]``, ``v_i ∈ [−v_max[i], v_max[i]]``,
    ``g_i ∈ [−g_max[i], g_max[i]]^{gdim}``. ``v_max`` / ``g_max`` are per-mode
    arrays of length ``n_modes``."""
    dim_F = gdim * gdim
    dim_v = n_modes
    dim_g = n_modes * gdim
    dim = dim_F + dim_v + dim_g
    v_max = np.asarray(v_max).reshape(n_modes)
    g_max = np.asarray(g_max).reshape(n_modes)

    if sampler in ("lhs", "sobol"):
        from scipy.stats import qmc
        engine = (qmc.LatinHypercube(d=dim, seed=seed) if sampler == "lhs"
                  else qmc.Sobol(d=dim, seed=seed))
        unit = engine.random(n_samples)
    elif sampler == "uniform":
        unit = np.random.default_rng(seed).random((n_samples, dim))
    else:
        raise ValueError(f"sampler must be 'lhs', 'sobol' or 'uniform', got {sampler!r}")

    s = 2.0 * unit - 1.0   # → [-1, 1]
    eye = np.eye(gdim)
    samples = []
    for row in s:
        Fbar = eye + row[:dim_F].reshape(gdim, gdim) * max_strain
        v = row[dim_F:dim_F + dim_v] * v_max
        g = row[dim_F + dim_v:].reshape(n_modes, gdim) * g_max[:, None]
        samples.append((Fbar, v, g))
    return samples


_T_RE = re.compile(r"^(?P<field>.+?)_(?P<t>[0-9]+\.[0-9]+)\.npy$")


def _merge_and_summarize(worker_root, pool_dir, samples, gdim):
    """Flatten per-worker snapshots into ``pool_dir`` and compute the realized
    ``(F̄, v, g)`` ranges from the converged snapshots.

    Each saved snapshot at load parameter ``t`` corresponds to the ramped state
    ``(I + t(F̄−I), t·v, t·g)``, so the realized ranges follow from the sample
    targets and the set of ``t`` actually reached (= the steps that converged)."""
    if os.path.isdir(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)

    eye = np.eye(gdim)
    n_copied = 0
    realized_F, realized_v, realized_g = [], [], []

    for sample_dir in sorted(glob(os.path.join(worker_root, "sample_*"))):
        try:
            idx = int(os.path.basename(sample_dir).split("_")[-1])
        except ValueError:
            continue
        snap_dir = os.path.join(sample_dir, "snapshots")
        if not os.path.isdir(snap_dir):
            continue
        Fbar_t, v_t, g_t = samples[idx]
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
                realized_v.append(t * v_t)
                realized_g.append(t * g_t)

    def _range(arr_list, shape):
        if not arr_list:
            return (np.full(shape, np.nan), np.full(shape, np.nan))
        a = np.stack(arr_list)
        return (a.min(axis=0), a.max(axis=0))

    realized = {
        "Fbar": _range(realized_F, (gdim, gdim)),
        "v": _range(realized_v, (len(samples[0][1]),)),
        "g": _range(realized_g, samples[0][2].shape),
    }
    return n_copied, realized


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_training_data(
    mesh_path: str, comm, gdim: int, material, max_strain: float, *,
    lattice_vectors: "np.ndarray | None" = None,
    degree: int = 2,
    modes_dir: "str | None" = None,
    n_modes: "int | None" = None,
    n_samples: int = 64,
    amplitude_factor: float = 2.0,
    sampler: str = "lhs",
    seed: int = 12345,
    n_workers: "int | None" = None,
    check_stability: bool = False,
    newton_options: "dict | None" = None,
    timestepper_options: "dict | None" = None,
    save_fields: "tuple | list" = _DEFAULT_SAVE_FIELDS,
    rve_volume: "float | None" = None,
    output_dir: "str | None" = None,
    extract_kwargs: "dict | None" = None,
) -> TrainingData:
    """Generate micromorphic ROM training data for an RVE.

    Parameters
    ----------
    mesh_path, comm, gdim, material : the RVE (as for ``mm.MicroSolver``). The
        driver runs as a single process and uses ``MPI.COMM_SELF`` internally;
        ``comm`` is only used for the (optional) mode-extraction phase, so run
        this with plain ``python`` (the pool provides the parallelism), not
        ``mpirun``.
    max_strain : the intended maximum deformation (per-component half-width of
        the ``F̄`` box and the scale for the ``v``/``g`` bounds).
    lattice_vectors : ``None`` for an axis-aligned box, ``(gdim, gdim)`` for a
        periodic polygon (forwarded to the solver and to mode extraction).
    modes_dir : where φ are cached / written (default ``output_dir/modes``).
        Reused across runs — delete it to force re-extraction.
    n_modes : keep only the first ``n_modes`` φ (default: all available).
    n_samples : number of ``(F̄, v, g)`` samples.
    amplitude_factor : the generosity factor ``κ`` in
        ``v_max[i] = κ·max_strain·L_RVE / ⟨‖φ_i‖⟩``. Bias it high — overshoots
        just fail to converge and are discarded.
    sampler : ``"lhs"`` (default), ``"sobol"`` or ``"uniform"``.
    seed, n_workers : RNG seed and pool size (default ``cpu_count − 1``).
    check_stability : monitor RVE stability during the sample solves. Default
        ``False`` — we want the equilibrium fluctuation at the *prescribed*
        ``(F̄, v, g)`` (the buckling component is carried by ``v``, not ``w``).
    newton_options, timestepper_options : inner-solve options (sensible defaults).
    save_fields : snapshot fields to save (default ``("u_fluc", "P")`` — what the
        micromorphic ``build_rom.py`` consumes). The sample solves are pure
        forward problems: no effective quantities / tangents are evaluated and no
        macro-sensitivity (adjoint) solves are done.
    rve_volume : exact cell volume ``|Q|`` (required for porous polygon cells).
    output_dir : root for ``modes/``, ``snapshots_pool/``, ``workers/`` and the
        summary ``.npz`` files (default ``./mm_training_output``).
    extract_kwargs : forwarded to :func:`extract_buckling_modes` if φ must be
        built (e.g. ``{"strategy": "lba", "max_strain": 0.1}``).

    Returns
    -------
    TrainingData with the bounds used, the realized ``(F̄, v, g)`` ranges of the
    converged snapshots, and the paths consumed by ``build_rom.py``.
    """
    if comm.size > 1:
        logger.warning("generate_training_data is a single-process driver "
                       "(multiprocessing pool); comm.size=%d — run with plain "
                       "`python`, not mpirun.", comm.size)

    output_dir = output_dir or os.path.abspath("mm_training_output")
    modes_dir = modes_dir or os.path.join(output_dir, "modes")
    pool_dir = os.path.join(output_dir, "snapshots_pool")
    worker_root = os.path.join(output_dir, "workers")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Enrichment modes φ (build if absent).
    phi_arrays = _ensure_modes(
        mesh_path, comm, gdim, material, degree=degree,
        lattice_vectors=lattice_vectors, modes_dir=modes_dir,
        n_modes=n_modes, extract_kwargs=extract_kwargs,
    )
    n_modes = len(phi_arrays)

    # 2. Bounds from L_RVE + max_strain (generous; convergence filters the rest).
    #    Per mode, v_max = κ·ε·L / ⟨‖φ_i‖⟩ so the bound tracks the *physical*
    #    pattern displacement regardless of how φ is normalised.
    L_rve = _rve_length_scale(mesh_path, gdim)
    mean_norms = _phi_mean_norms(mesh_path, gdim, degree, phi_arrays)
    v_max = amplitude_factor * max_strain * L_rve / mean_norms
    g_max = v_max / L_rve
    bounds = {"max_strain": max_strain, "L_RVE": L_rve,
              "amplitude_factor": amplitude_factor, "phi_mean_norms": mean_norms,
              "v_max": v_max, "g_max": g_max}
    with np.printoptions(precision=4, suppress=True):
        logger.info("Bounds: L_RVE=%.4g, max_strain=%.4g, κ=%.2g, ⟨‖φ‖⟩=%s",
                    L_rve, max_strain, amplitude_factor, mean_norms)
        logger.info("        → v_max=%s, g_max=%s", v_max, g_max)

    # 3. Space-filling samples.
    samples = _draw_samples(n_samples, n_modes, gdim, max_strain, v_max, g_max,
                            seed, sampler)

    # 4. Parallel FOM solves (multiprocessing pool, COMM_SELF per worker).
    if worker_root and os.path.isdir(worker_root):
        shutil.rmtree(worker_root)
    os.makedirs(worker_root, exist_ok=True)
    cfg = _WorkerConfig(
        mesh_path=mesh_path, gdim=gdim, degree=degree, material=material,
        n_modes=n_modes, phi_arrays=phi_arrays, lattice_vectors=lattice_vectors,
        rve_volume=rve_volume, check_stability=check_stability,
        save_fields=tuple(save_fields),
        newton_options=newton_options or _DEFAULT_NEWTON,
        timestepper_options=timestepper_options or _DEFAULT_TIMESTEP,
        worker_root=worker_root,
    )
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    tasks = [(i, Fbar, v, g) for i, (Fbar, v, g) in enumerate(samples)]
    logger.info("Running %d sample(s) on %d worker(s) (N=%d mode(s)) …",
                n_samples, n_workers, n_modes)
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

    # 5. Merge + realized ranges + side-tables.
    n_copied, realized = _merge_and_summarize(worker_root, pool_dir, samples, gdim)
    targets = {
        "Fbar": np.stack([s[0] for s in samples]),
        "v": np.stack([s[1] for s in samples]),
        "g": np.stack([s[2] for s in samples]),
        "ok": ok_flags,
    }
    np.savez(os.path.join(output_dir, "sample_inputs.npz"), **targets)
    np.savez(os.path.join(output_dir, "training_summary.npz"),
             v_max=v_max, g_max=g_max, L_RVE=L_rve, max_strain=max_strain,
             amplitude_factor=amplitude_factor,
             realized_Fbar_lo=realized["Fbar"][0], realized_Fbar_hi=realized["Fbar"][1],
             realized_v_lo=realized["v"][0], realized_v_hi=realized["v"][1],
             realized_g_lo=realized["g"][0], realized_g_hi=realized["g"][1])

    logger.info("Pooled %d snapshot file(s) → %s", n_copied, pool_dir)
    with np.printoptions(precision=4, suppress=True):
        logger.info("Realized F̄ range (converged): [%s,\n %s]",
                    realized["Fbar"][0], realized["Fbar"][1])
        logger.info("Realized v range: [%s, %s]", realized["v"][0], realized["v"][1])
        logger.info("Realized g range: [%s, %s]",
                    realized["g"][0].ravel(), realized["g"][1].ravel())

    return TrainingData(
        output_dir=output_dir, pool_dir=pool_dir, modes_dir=modes_dir,
        n_modes=n_modes, n_samples=n_samples, n_converged=n_converged,
        bounds=bounds, realized_ranges=realized, targets=targets,
    )
