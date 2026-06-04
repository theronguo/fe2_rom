"""Random (F̄, v, g) sampling + parallel FOM snapshot collection.

Drives :class:`MicroSolver` (N=1 mode) over
``N_SAMPLES`` random macro states drawn uniformly around the FOM demo values
in ``run_micromorphic.py``, using a ``multiprocessing.Pool`` of workers.

Pipeline:

1. **Phi precomputation (once).** A linear buckling analysis at the reference
   state populates ``output/snapshots/phi_*.npy``. Skipped if those files
   already exist.
2. **Sampling.** ``N_SAMPLES`` triples ``(F̄, v, g)`` drawn from uniform
   intervals (configurable at the top of this file). Seeded for reproducibility.
3. **Worker pool.** Each worker:
     * builds a solver on ``MPI.COMM_SELF`` (each process is independent),
     * injects the precomputed φ into ``solver._phi`` and calls
       ``rebuild_constraints()`` so the projected Newton sees the right
       ⟨w·φ⟩ / ⟨(w·φ)X⟩ rows,
     * runs ``solver(F̄, v, g)`` which writes ``u_fluc_*.npy`` and ``P_*.npy``
       into the worker's private ``output/workers/sample_NNNN/snapshots/``.
4. **Merge.** Worker snapshots are renamed into a single flat directory
   ``output/snapshots_pool/`` with names ``{field}_s{sample:04d}_{t:.5f}.npy``
   suitable for ``build_rom.py``'s glob.

Run:
    python sample_micromorphic.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import multiprocessing as mp
import re
import shutil
from pathlib import Path

import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import NeoHookean, setup_logging
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.mm.microsolver import MicroSolver


# --- Configuration ----------------------------------------------------------
HERE = Path(__file__).parent
RVE_MESH = str(HERE / "rve.msh")

OUTPUT_ROOT = HERE / "output"
PHI_DIR = OUTPUT_ROOT / "snapshots"                # consumed by build_rom.py
POOL_DIR = OUTPUT_ROOT / "snapshots_pool"           # flat merged snapshots
WORKER_ROOT = OUTPUT_ROOT / "workers"               # per-worker scratch dirs

N_MODES = 1
N_SAMPLES = 16
N_WORKERS = 8 #max(1, mp.cpu_count() - 1)
RNG_SEED = 12345

# Uniform sampling bounds around the FOM demo values.
FBAR_DIAG_RANGE    = (0.85, 1.05)
FBAR_OFFDIAG_RANGE = (-0.05, 0.05)
V_RANGE = (-1.0, 1.0)
G_RANGE = (-0.2, 0.2)

# Material (matches run_micromorphic.py).
E_MICRO, NU_MICRO = 3000.0, 0.30
MU_MICRO  = E_MICRO / (2.0 * (1.0 + NU_MICRO))
LAM_MICRO = E_MICRO * NU_MICRO / ((1.0 + NU_MICRO) * (1.0 - 2.0 * NU_MICRO))

GDIM = 2
DEGREE = 2

NEWTON_OPTS = {
    "rel_tol": 1e-8, "abs_tol": 1e-6,
    "max_iter": 50, "div_rel_tol": 10,
    "switch_to_minres": True,
}
TIMESTEP_OPTS = {
    "t_end": 1.0, "dt_init": 1e-2, "dt_min": 1e-5,
    "dt_max": 1e-2, "good_newton_steps": 5,
}


# --- Helpers ----------------------------------------------------------------

def _make_material():
    return NeoHookean(mu=MU_MICRO, lmbda=LAM_MICRO)


def _sample_inputs(rng: np.random.Generator):
    Fbar = np.eye(GDIM)
    Fbar[0, 0] = rng.uniform(*FBAR_DIAG_RANGE)
    Fbar[1, 1] = rng.uniform(*FBAR_DIAG_RANGE)
    Fbar[0, 1] = rng.uniform(*FBAR_OFFDIAG_RANGE)
    Fbar[1, 0] = rng.uniform(*FBAR_OFFDIAG_RANGE)
    v = rng.uniform(V_RANGE[0], V_RANGE[1], size=N_MODES)
    g = rng.uniform(G_RANGE[0], G_RANGE[1], size=(N_MODES, GDIM))
    return Fbar, v, g


def _phi_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.glob("phi_*.npy")
        if re.match(r"phi_[\d.]+\.npy$", p.name)
    )


# --- Phi precomputation -----------------------------------------------------

def precompute_phi() -> list[np.ndarray]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = _phi_files(PHI_DIR)
    if len(existing) >= N_MODES:
        print(f"[phi] using existing {len(existing)} mode(s) in {PHI_DIR}")
        return [np.load(p) for p in existing[:N_MODES]]

    print("[phi] computing linear buckling modes (reference state F̄=I) ...")
    setup_logging(MPI.COMM_SELF, level=logging.WARNING)
    solver = MicroSolver(
        mesh_path=RVE_MESH, comm=MPI.COMM_SELF, gdim=GDIM,
        material=_make_material(), N=N_MODES,
        degree=DEGREE, output_dir=str(OUTPUT_ROOT),
        check_stability=False,
        visualize_fields=[],
        newton_options=NEWTON_OPTS,
        timestepper_options=TIMESTEP_OPTS,
        averages_only_final=True,
    )
    solver.compute_linear_buckling_modes(N_MODES, save_modes=True)
    files = _phi_files(PHI_DIR)
    if len(files) < N_MODES:
        raise RuntimeError(
            f"compute_linear_buckling_modes produced {len(files)} files, "
            f"expected {N_MODES}"
        )
    return [np.load(p) for p in files[:N_MODES]]


# --- Worker -----------------------------------------------------------------

def _worker_task(args):
    sample_idx, Fbar, v, g, phi_arrays = args

    print(f"[worker {os.getpid():04d}] starting sample {sample_idx}")
    # Make sure the per-worker thread caps stick (some BLAS backends only
    # honour env vars set at import; safe to re-set defensively).
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    work_dir = WORKER_ROOT / f"sample_{sample_idx:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Each worker runs on COMM_SELF — no inter-process MPI communication.
    setup_logging(MPI.COMM_SELF, level=logging.ERROR)
    solver = MicroSolver(
        mesh_path=RVE_MESH, comm=MPI.COMM_SELF, gdim=GDIM,
        material=_make_material(), N=N_MODES,
        degree=DEGREE, output_dir=str(work_dir),
        check_stability=True,
        # ``visualize_fields`` must include "P"/"A" so the solver allocates
        # ``self.P_func``/``self.A_func`` — ``save_snapshots`` reads them.
        # "A" is rank-4 so it is not written to the .bp file, only saved as
        # .npy snapshots. The per-worker .bp file is harmless overhead.
        visualize_fields=["P"],
        save_snapshots=["u_fluc", "P", "A", "dw_dFbar", "dw_dv", "dw_dg"],
        newton_options=NEWTON_OPTS,
        timestepper_options=TIMESTEP_OPTS,
        averages_only_final=False,
    )

    # Inject precomputed phi and refresh constraint vectors.
    for i, arr in enumerate(phi_arrays):
        if solver._phi[i].x.array.size != arr.size:
            return (sample_idx, False,
                    f"phi[{i}] size mismatch: "
                    f"solver={solver._phi[i].x.array.size} disk={arr.size}")
        solver._phi[i].x.array[:] = arr
        solver._phi[i].x.scatter_forward()
    solver.rebuild_constraints()

    try:
        solver(Fbar, v, g)
        return (sample_idx, True, "")
    except RVEConvergenceError as e:
        return (sample_idx, False, f"RVEConvergenceError: {e}")
    except Exception as e:  # noqa: BLE001
        return (sample_idx, False, f"{type(e).__name__}: {e}")


# --- Merge ------------------------------------------------------------------

_T_RE = re.compile(r"^(?P<field>.+?)_(?P<t>[\d.]+)\.npy$")


def _merge_worker_snapshots() -> int:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale pool contents.
    for p in POOL_DIR.iterdir():
        if p.is_file():
            p.unlink()

    n_copied = 0
    for sample_dir in sorted(WORKER_ROOT.iterdir()):
        snap_dir = sample_dir / "snapshots"
        if not snap_dir.is_dir():
            continue
        try:
            idx = int(sample_dir.name.split("_")[-1])
        except ValueError:
            continue
        for f in sorted(snap_dir.glob("*.npy")):
            # Carry per-field DOF-coord files over once (shared across samples).
            if f.name.endswith("_dof_coords.npy"):
                tgt = POOL_DIR / f.name
                if not tgt.exists():
                    shutil.copy(f, tgt)
                continue
            m = _T_RE.match(f.name)
            if not m:
                continue
            field, t_str = m.group("field"), m.group("t")
            if field == "phi":
                continue  # phi lives in PHI_DIR; not a snapshot
            tgt = POOL_DIR / f"{field}_s{idx:04d}_{t_str}.npy"
            shutil.copy(f, tgt)
            n_copied += 1
    return n_copied


# --- Main -------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    phi_arrays = precompute_phi()
    print(f"[phi] loaded {len(phi_arrays)} mode(s), shape={phi_arrays[0].shape}")

    rng = np.random.default_rng(RNG_SEED)
    tasks = []
    for i in range(N_SAMPLES):
        Fbar, v, g = _sample_inputs(rng)
        tasks.append((i, Fbar, v, g, phi_arrays))

    if WORKER_ROOT.exists():
        shutil.rmtree(WORKER_ROOT)
    WORKER_ROOT.mkdir(parents=True)

    print(f"[pool] launching N_SAMPLES={N_SAMPLES}, N_WORKERS={N_WORKERS}")
    # spawn start: avoid inheriting any open MPI / FFCX state from the parent.
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        results = pool.map(_worker_task, tasks, chunksize=1)

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = N_SAMPLES - n_ok
    print(f"[pool] done: {n_ok} ok, {n_fail} failed.")
    if n_fail:
        for idx, ok, msg in results:
            if not ok:
                print(f"  sample {idx:04d}: {msg}")

    # Sampled inputs side-table for downstream analysis.
    np.savez(
        OUTPUT_ROOT / "sample_inputs.npz",
        sample_idx=np.array([t[0] for t in tasks]),
        Fbar=np.stack([t[1] for t in tasks]),
        v=np.stack([t[2] for t in tasks]),
        g=np.stack([t[3] for t in tasks]),
        ok=np.array([ok for _, ok, _ in results], dtype=bool),
    )

    n_copied = _merge_worker_snapshots()
    print(f"[merge] {n_copied} snapshot files -> {POOL_DIR}")
    print(f"        sample inputs            -> {OUTPUT_ROOT / 'sample_inputs.npz'}")


if __name__ == "__main__":
    main()
