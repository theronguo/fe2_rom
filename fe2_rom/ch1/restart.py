"""Checkpoint / restart helpers for the full two-scale CH1 macro solver.

The macro driver writes one checkpoint per accepted load step (overwriting
the previous one atomically) and reloads it automatically on the next
invocation when the same ``output_dir`` is reused.

Layout under ``output_dir``::

    checkpoint/
    ├── meta.json            # t_current, dt, n_ranks, gdim, schema, ...
    ├── reaction.npz         # displacements[], forces[]   (rank 0)
    ├── macro_state.npz      # rank-keyed: macro Function .x.array per rank
    └── rves/
        └── rank_{r}.npz     # per-rank stacked RVE state

Restart requires the **same MPI rank count** as the original run. Each
rank validates a partition fingerprint (sorted local cell-midpoint
coords) on load to catch any partitioning reshuffle.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any

import numpy as np
from mpi4py import MPI


SCHEMA_VERSION = 1

CKPT_NAME = "checkpoint"
TMP_NAME = "checkpoint.tmp"


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

def checkpoint_dirs(output_dir: str) -> tuple[str, str]:
    return os.path.join(output_dir, CKPT_NAME), os.path.join(output_dir, TMP_NAME)


def _rve_dir(base: str) -> str:
    return os.path.join(base, "rves")


def _rank_file(base: str, rank: int) -> str:
    return os.path.join(_rve_dir(base), f"rank_{rank}.npz")


def _macro_file(base: str) -> str:
    return os.path.join(base, "macro_state.npz")


def _meta_file(base: str) -> str:
    return os.path.join(base, "meta.json")


def _reaction_file(base: str) -> str:
    return os.path.join(base, "reaction.npz")


# ----------------------------------------------------------------------
# Fingerprint
# ----------------------------------------------------------------------

def compute_partition_fingerprint(mesh) -> str:
    """sha256 of sorted local cell midpoint coords + cell count.

    Cheap, rank-local, deterministic for a fixed partitioning. Used to
    detect partitioning reshuffles between the original and restarted
    runs.
    """
    tdim = mesh.topology.dim
    n_cells = mesh.topology.index_map(tdim).size_local
    if n_cells == 0:
        midpoints = np.zeros((0, mesh.geometry.dim), dtype=np.float64)
    else:
        from dolfinx import mesh as dmesh
        cells = np.arange(n_cells, dtype=np.int32)
        midpoints = dmesh.compute_midpoints(mesh, tdim, cells)
    # Sort rows for order independence within the rank.
    order = np.lexsort(midpoints.T[::-1])
    midpoints = midpoints[order]
    h = hashlib.sha256()
    h.update(np.int64(n_cells).tobytes())
    h.update(midpoints.astype(np.float64).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# Atomic finalize
# ----------------------------------------------------------------------

def prepare_tmp(comm, output_dir: str) -> str:
    ckpt_dir, tmp_dir = checkpoint_dirs(output_dir)
    if comm.rank == 0:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(_rve_dir(tmp_dir), exist_ok=True)
    comm.Barrier()
    return tmp_dir


def atomic_finalize(comm, output_dir: str) -> None:
    ckpt_dir, tmp_dir = checkpoint_dirs(output_dir)
    comm.Barrier()
    if comm.rank == 0:
        if os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.rename(tmp_dir, ckpt_dir)
    comm.Barrier()


# ----------------------------------------------------------------------
# Completeness check (collective)
# ----------------------------------------------------------------------

def checkpoint_complete(comm, output_dir: str) -> bool:
    ckpt_dir, _ = checkpoint_dirs(output_dir)
    if comm.rank == 0:
        ok = (
            os.path.isdir(ckpt_dir)
            and os.path.isfile(_meta_file(ckpt_dir))
            and os.path.isfile(_macro_file(ckpt_dir))
            and os.path.isfile(_reaction_file(ckpt_dir))
            and all(
                os.path.isfile(_rank_file(ckpt_dir, r))
                for r in range(comm.size)
            )
        )
        flag = 1 if ok else 0
    else:
        flag = 0
    flag = comm.bcast(flag, root=0)
    return bool(flag)


# ----------------------------------------------------------------------
# Meta json
# ----------------------------------------------------------------------

def save_meta(comm, base: str, **kw: Any) -> None:
    if comm.rank == 0:
        kw.setdefault("schema_version", SCHEMA_VERSION)
        kw.setdefault("n_ranks", comm.size)
        with open(_meta_file(base), "w") as f:
            json.dump(kw, f, indent=2)


def load_meta(comm, base: str) -> dict:
    if comm.rank == 0:
        with open(_meta_file(base)) as f:
            data = json.load(f)
    else:
        data = None
    return comm.bcast(data, root=0)


# ----------------------------------------------------------------------
# Reaction logger
# ----------------------------------------------------------------------

def save_reaction(comm, logger, base: str) -> None:
    if logger is None:
        return
    if comm.rank == 0:
        np.savez(
            _reaction_file(base),
            displacements=np.asarray(logger.displacements, dtype=np.float64),
            forces=np.asarray(logger.forces, dtype=np.float64),
        )


def load_reaction(comm, logger, base: str) -> None:
    if logger is None:
        return
    if comm.rank == 0:
        data = np.load(_reaction_file(base))
        disps = data["displacements"].tolist()
        forces = data["forces"].tolist()
    else:
        disps, forces = None, None
    disps = comm.bcast(disps, root=0)
    forces = comm.bcast(forces, root=0)
    logger.displacements = list(disps)
    logger.forces = list(forces)


# ----------------------------------------------------------------------
# Macro Function (per-rank dof array)
# ----------------------------------------------------------------------

def save_macro_field(comm, fn, base: str, fingerprint: str) -> None:
    """Save one macro Function as ``macro_state.npz`` with arrays keyed by
    ``rank_{r}`` (containing the local dof array including ghosts) and
    ``fp_{r}`` (the rank's partition fingerprint).

    Single-file gather is fine for the macro field — it's the small one
    in FE² (RVE state dominates disk).
    """
    local = fn.x.array.copy()
    all_local = comm.gather(local, root=0)
    all_fp = comm.gather(fingerprint, root=0)
    if comm.rank == 0:
        out = {}
        for r, (arr, fp) in enumerate(zip(all_local, all_fp)):
            out[f"rank_{r}"] = np.asarray(arr)
            out[f"fp_{r}"] = np.asarray(fp)
        np.savez(_macro_file(base), **out)


def load_macro_field(comm, fn, base: str, fingerprint: str) -> None:
    if comm.rank == 0:
        data = np.load(_macro_file(base))
        chunks = [data[f"rank_{r}"] for r in range(comm.size)]
        fps = [str(data[f"fp_{r}"]) for r in range(comm.size)]
    else:
        chunks, fps = None, None
    my_chunk = comm.scatter(chunks, root=0)
    my_fp = comm.scatter(fps, root=0)
    if my_fp != fingerprint:
        raise RuntimeError(
            f"Macro partition fingerprint mismatch on rank {comm.rank} — "
            "checkpoint was written with a different partitioning. Rerun "
            "with the same MPI rank count, or delete the checkpoint to "
            "start fresh."
        )
    if my_chunk.shape != fn.x.array.shape:
        raise RuntimeError(
            f"Macro dof array shape mismatch on rank {comm.rank}: "
            f"checkpoint has {my_chunk.shape}, expected {fn.x.array.shape}."
        )
    fn.x.array[:] = my_chunk
    fn.x.scatter_forward()


# ----------------------------------------------------------------------
# RVE state I/O (used by RVEMaterial.save_rves/load_rves)
# ----------------------------------------------------------------------

def rank_state_path(base: str, rank: int) -> str:
    return _rank_file(base, rank)


def write_rank_state(base: str, rank: int, fingerprint: str,
                     stacked: dict[str, np.ndarray]) -> None:
    os.makedirs(_rve_dir(base), exist_ok=True)
    payload = {"fingerprint": np.asarray(fingerprint)}
    payload.update(stacked)
    np.savez(_rank_file(base, rank), **payload)


def read_rank_state(base: str, rank: int) -> tuple[str, dict[str, np.ndarray]]:
    data = np.load(_rank_file(base, rank))
    fp = str(data["fingerprint"])
    state = {k: data[k] for k in data.files if k != "fingerprint"}
    return fp, state


# ----------------------------------------------------------------------
# Optional full-trajectory dumps (independent of the rolling checkpoint)
# ----------------------------------------------------------------------

def macro_snapshot_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "snapshots")


def rve_history_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "rve_history")


def save_macro_snapshot(comm, fn, output_dir: str, step_index: int,
                        t: float, fingerprint: str) -> None:
    """Append a per-rank gather of ``fn.x.array`` to
    ``output_dir/snapshots/macro_step_{step_index:06d}.npz``.

    Skips silently on rank > 0 — rank 0 writes the gathered file.
    """
    local = fn.x.array.copy()
    all_local = comm.gather(local, root=0)
    all_fp = comm.gather(fingerprint, root=0)
    if comm.rank == 0:
        d = macro_snapshot_dir(output_dir)
        os.makedirs(d, exist_ok=True)
        out = {"t": np.float64(t), "step_index": np.int64(step_index)}
        for r, (arr, fp) in enumerate(zip(all_local, all_fp)):
            out[f"rank_{r}"] = np.asarray(arr)
            out[f"fp_{r}"] = np.asarray(fp)
        np.savez(
            os.path.join(d, f"macro_step_{step_index:06d}.npz"),
            **out,
        )


def quadrature_point_info(comm, qmap, gather: bool = True):
    """Return the physical coordinates of every macro quadrature point.

    Parameters
    ----------
    gather : bool
        If True (default), rank 0 returns a list of dicts
        ``{"rank": r, "qp": i, "x": [x, y, (z)]}`` covering **all** qps
        across all ranks (other ranks return an empty list). Useful for
        deciding which ``rve_history_qps`` to flag interactively.
        If False, every rank returns its own list of dicts (with its own
        ``rank`` tag and **rank-local** ``qp`` index).
    """
    fn = next(iter(qmap.fluxes.values()))
    V = fn.function_space
    coords_all = V.tabulate_dof_coordinates()
    n_local = V.dofmap.index_map.size_local
    local_coords = coords_all[:n_local]
    rank = comm.rank
    local = [
        {"rank": rank, "qp": i, "x": local_coords[i].tolist()}
        for i in range(n_local)
    ]
    if not gather:
        return local
    gathered = comm.gather(local, root=0)
    if rank == 0:
        flat = []
        for chunk in gathered:
            flat.extend(chunk)
        return flat
    return []


def save_rve_history(rank: int, output_dir: str, step_index: int, t: float,
                     rve_states: dict[int, dict[str, np.ndarray]]) -> None:
    """For each ``(qp_local_index, state_dict)`` in ``rve_states`` write
    one ``rank_{r}_qp_{i}_step_{N:06d}.npz`` file. Caller is responsible
    for selecting which RVEs to dump.
    """
    if not rve_states:
        return
    d = rve_history_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    for qp, state in rve_states.items():
        payload = {"t": np.float64(t), "step_index": np.int64(step_index)}
        payload.update(state)
        np.savez(
            os.path.join(d, f"rank_{rank}_qp_{qp}_step_{step_index:06d}.npz"),
            **payload,
        )


def qp_history_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "qp_history")


def save_qp_history(comm, qmap, output_dir: str, step_index: int,
                    t: float) -> None:
    """Write the macro gradients at every *owned* quadrature point on this
    rank to ``output_dir/qp_history/rank_{r}_step_{step_index:06d}.npz``.

    Values are read straight off the registered qmap gradients (re-evaluated
    on the current solution), so this works for any constitutive law
    (FOM / ROM / dummy). Per-qp layout — cell-major, aligned with
    :func:`quadrature_point_info`:

        x : (n_qp, gdim)     quadrature-point physical coordinates
        F : (n_qp, F_dim)    ``nonsymmetric_tensor_to_vector(I + grad u)``
        v : (n_qp, N)        enrichment amplitudes        (omitted if N == 0)
        g : (n_qp, N*gdim)   grad of enrichment, mode-major (omitted if N == 0)

    Reconstruct the F matrix and reshape g downstream with
    ``fe2_rom.mm.material._fvec_to_mat`` / ``g.reshape(N, gdim)``.
    """
    flux_fn = next(iter(qmap.fluxes.values()))
    Vq = flux_fn.function_space
    n_local = Vq.dofmap.index_map.size_local
    coords = Vq.tabulate_dof_coordinates()[:n_local]

    payload: dict[str, np.ndarray] = {
        "t": np.float64(t),
        "step_index": np.int64(step_index),
        "x": np.asarray(coords),
        "qp_local_index": np.arange(n_local, dtype=np.int64),
    }
    payload["F"] = np.asarray(
        qmap.get_gradient_vals(qmap.gradients["F"], qmap.cells)[:n_local]
    )
    if "v" in qmap.gradients:
        payload["v"] = np.asarray(
            qmap.get_gradient_vals(qmap.gradients["v"], qmap.cells)[:n_local]
        )
        payload["g"] = np.asarray(
            qmap.get_gradient_vals(qmap.gradients["g"], qmap.cells)[:n_local]
        )

    d = qp_history_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    np.savez(
        os.path.join(d, f"rank_{comm.rank}_step_{step_index:06d}.npz"),
        **payload,
    )
