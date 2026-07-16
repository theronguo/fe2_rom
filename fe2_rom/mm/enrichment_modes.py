"""Extract micromorphic enrichment modes ``φ`` from full-order buckling of an RVE.

This is *Problem 1* of the micromorphic-ROM workflow: given only a mesh and a
material, find a good set of global modes ``φᵢ`` for the micromorphic ansatz

    u_total = (F̄ − I)·X + Σᵢ (vᵢ + X·gᵢ) φᵢ + w

A linear buckling analysis at the *undeformed* state generally gives the wrong
modes, so we drive the RVE into buckling under compression and learn the modes
from the result. One public entry point does everything:

    >>> from fe2_rom.mm.enrichment_modes import extract_buckling_modes
    >>> result = extract_buckling_modes(mesh_path, comm, gdim, material,
    ...                                 lattice_vectors=..., output_dir="output")
    >>> result.phi          # (n_dofs, N) POD modes; result.n_modes == N

Pipeline (per "end deformation" F̄_target, then pooled):

  1. Run a full-order first-order (CH1) periodic homogenization solve under
     compression (``fe2_rom.ch1.MicroSolver``, stability monitoring on).
  2. Obtain the buckling fields by one of two ``strategy`` choices:
       * ``"traverse"``  – perturb onto the buckled branch and trace it; keep the
         post-buckling snapshots, detected by the jump in ``||u_fluc||`` (also
         saves the matching stress ``P`` as g=0 ROM training data). If the branch
         can't be traversed, fall back to LBA-at-critical.
       * ``"lba"``       – do **not** attempt to traverse: step as close to the
         bifurcation as possible (perturbation disabled, dt halved at the
         instability) and do a linear buckling analysis there. Cleaner/faster for
         symmetric (degenerate) loadings that don't traverse anyway.
  3. Pool the modes/snapshots over all deformations, compute an H¹-orthonormal
     POD basis, and take ``N`` at the first large drop in the singular-value
     spectrum (degenerate clusters kept whole).

The default ``end_deformations`` is a single equal-biaxial (2D) / equal-triaxial
(3D) compression: the symmetric loading captures the full degenerate buckling
subspace (all symmetry-related orientations) in one shot.

Geometry-agnostic: pass ``lattice_vectors`` for a polygonal RVE (e.g. a hexagon)
or leave it ``None`` for an axis-aligned box.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from glob import glob

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io

from fe2_rom.ch1.microsolver import MicroSolver
from fe2_rom.ch1.exceptions import RVEConvergenceError
from fe2_rom.hyperelastic_solver.logging_utils import silence_c_stdout
from fe2_rom.rom.pod import POD

logger = logging.getLogger("fe2_rom.mm.enrichment_modes")

_DEFAULT_NEWTON = {
    "rel_tol": 1e-8, "abs_tol": 1e-6, "max_iter": 10, "max_iter_instab": 30,
    "div_rel_tol": 10.0, "switch_to_minres": True,
}
_DEFAULT_STABILITY = {"nev": 6, "neg_tol": -1e-10}


# ---------------------------------------------------------------------------
# Public result + helpers
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentModes:
    """Output of :func:`extract_buckling_modes`."""
    phi: np.ndarray             # (n_dofs, N) POD modes (columns)
    singular_values: np.ndarray
    n_modes: int
    dof_coords: np.ndarray      # (n_nodes, gdim) for phi / u_fluc
    snapshots_u: np.ndarray     # (n_snap, n_dofs) pooled post-buckling u_fluc
    snapshots_P: np.ndarray     # (n_snap, n_P) pooled post-buckling stress (may be empty)


def make_symmetric_compression(max_strain: float, gdim: int) -> list[np.ndarray]:
    """A single equal-biaxial (2D) / equal-triaxial (3D) compression target
    ``F̄ = (1 − max_strain) I``. The symmetric loading preserves the
    microstructure point group, so its buckling modes span the full degenerate
    subspace (all symmetry-related pattern orientations)."""
    return [(1.0 - max_strain) * np.eye(gdim)]


# ---------------------------------------------------------------------------
# Internal run configuration
# ---------------------------------------------------------------------------

@dataclass
class _RunConfig:
    mesh_path: str
    comm: object
    gdim: int
    material: object
    degree: int
    quadrature_degree: "int | None"
    lattice_vectors: "np.ndarray | None"
    work_dir: str
    gap_ratio: float
    pert_amplitude: float
    postbuckle_frac: float
    buckle_amp_ratio: float
    fallback_n_modes: int
    newton_options: dict
    stability_options: dict
    timestepper_options: dict


def _build_solver(cfg: _RunConfig, output_dir: str, *, check_stability: bool,
                  perturb_post_buckling: bool = True,
                  save_snapshots: bool = False) -> MicroSolver:
    # "P" must be in visualize_fields for the P snapshot path to find P_func.
    return MicroSolver(
        mesh_path=cfg.mesh_path, comm=cfg.comm, gdim=cfg.gdim,
        material=cfg.material, degree=cfg.degree,
        quadrature_degree=cfg.quadrature_degree, output_dir=output_dir,
        check_stability=check_stability,
        perturb_post_buckling=perturb_post_buckling,
        visualize_fields=["P"] if save_snapshots else [],
        save_snapshots=["u_fluc", "P"] if save_snapshots else [],
        newton_options=dict(cfg.newton_options),
        timestepper_options=dict(cfg.timestepper_options),
        stability_options=dict(cfg.stability_options),
        lattice_vectors=cfg.lattice_vectors,
    )


# ---------------------------------------------------------------------------
# Snapshot IO + mode selection
# ---------------------------------------------------------------------------

def _load_snapshots(snap_dir: str, field: str) -> tuple[list[np.ndarray], list[float]]:
    """Load ``{field}_<t>.npy`` snapshots sorted by load parameter t."""
    items: list[tuple[float, str]] = []
    for f in glob(os.path.join(snap_dir, f"{field}_*.npy")):
        m = re.search(rf"{field}_([0-9]+\.[0-9]+)\.npy$", os.path.basename(f))
        if m:
            items.append((float(m.group(1)), f))
    items.sort()
    return [np.load(f) for _, f in items], [t for t, _ in items]


def _ascending_gap_count(eigvals: np.ndarray, gap_ratio: float) -> int:
    """Number of soft (buckling-relevant) modes in an ascending eigenvalue
    spectrum: those before the **largest** upward jump, which separates the
    near-zero buckling cluster from the stiff modes. The largest gap is used
    (not the first) because the soft cluster itself can have internal
    sub-splittings — e.g. a hexagon's degenerate triplet splits into an E-pair +
    a slightly stiffer mode (~7×), far smaller than the soft/stiff gap (~1e5×).
    ``gap_ratio`` is a floor: if even the largest jump is below it, keep all."""
    e = np.abs(np.asarray(eigvals, dtype=float))
    e = e[np.isfinite(e)]
    if e.size <= 1:
        return int(e.size)
    ratios = e[1:] / np.maximum(e[:-1], 1e-30)
    k = int(np.argmax(ratios))
    return (k + 1) if ratios[k] >= gap_ratio else int(e.size)


def _n_modes_by_gap(singular_values: np.ndarray, gap_ratio: float) -> int:
    """First large *drop* in the descending POD singular-value spectrum: keep the
    modes before the first ``σ_i/σ_{i+1} ≥ gap_ratio`` (degenerate clusters whole)."""
    sv = np.asarray(singular_values, dtype=float)
    sv = sv[sv > sv[0] * 1e-12] if sv.size else sv
    if sv.size <= 1:
        return int(sv.size)
    ratios = sv[:-1] / np.maximum(sv[1:], 1e-30)
    for k, ratio in enumerate(ratios):
        if ratio >= gap_ratio:
            return k + 1
    return int(sv.size)


def _lba_modes(cfg: _RunConfig, F_ref: np.ndarray, tag: str, *,
               solver=None) -> list[np.ndarray]:
    """Linear buckling modes at a (near-critical) pre-stress ``F_ref`` — compute
    ``fallback_n_modes`` eigenpairs and keep the buckling-relevant ones (below the
    first eigenvalue gap).

    If ``solver`` is given it must already hold the converged equilibrium at
    ``F_ref`` — the spectrum is then assembled in place (``Fbar=None``, no
    re-solve). Otherwise a fresh solver is built and ramped to ``F_ref``."""
    if solver is None:
        solver = _build_solver(cfg, os.path.join(cfg.work_dir, f"{tag}_lba"),
                               check_stability=False)
        eigvals, modes = solver.compute_buckling_spectrum(
            cfg.fallback_n_modes, Fbar=F_ref, return_modes=True,
        )
    else:
        eigvals, modes = solver.compute_buckling_spectrum(
            cfg.fallback_n_modes, Fbar=None, return_modes=True,
        )
    n_keep = _ascending_gap_count(eigvals, cfg.gap_ratio)
    logger.info("  LBA eigenvalues %s → keep %d buckling mode(s)",
                np.array2string(eigvals, precision=4), n_keep)
    return [m.x.array.copy() for m in modes[:n_keep]]


# ---------------------------------------------------------------------------
# The two strategies for one load path
# ---------------------------------------------------------------------------

def _run_traverse(cfg: _RunConfig, target_F: np.ndarray,
                  tag: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Perturb onto the buckled branch and keep the post-buckling (u_fluc, P)
    snapshots, detected by the ``||u_fluc||`` jump. Fall back to LBA-at-critical
    if no buckling is observed."""
    path_dir = os.path.join(cfg.work_dir, f"path_{tag}")
    snap_dir = os.path.join(path_dir, "snapshots")
    if os.path.isdir(snap_dir):
        shutil.rmtree(snap_dir)
    os.makedirs(snap_dir, exist_ok=True)

    solver = _build_solver(cfg, path_dir, check_stability=True,
                           perturb_post_buckling=True, save_snapshots=True)
    try:
        solver(target_F, pert_amplitude_init=cfg.pert_amplitude)
    except RVEConvergenceError as exc:
        logger.warning("  stopped early (%s); using snapshots collected so far", exc)

    u_snaps, u_ts = _load_snapshots(snap_dir, "u_fluc")
    P_snaps, _ = _load_snapshots(snap_dir, "P")
    if not u_snaps:
        logger.warning("  no snapshots saved"); return [], []

    amps = np.array([np.linalg.norm(s) for s in u_snaps])
    baseline = float(np.median(amps[:max(1, len(amps) // 5)]))
    amax = float(amps.max())
    if amax >= cfg.buckle_amp_ratio * baseline:
        thr = cfg.postbuckle_frac * amax
        mask = amps >= thr
        u_keep = [s for s, m in zip(u_snaps, mask) if m]
        P_keep = ([s for s, m in zip(P_snaps, mask) if m]
                  if len(P_snaps) == len(u_snaps) else [])
        t_on = next(t for t, m in zip(u_ts, mask) if m)
        logger.info("  buckled: ||u_fluc|| %.3g → %.3g (×%.0f); %d post-buckling "
                    "snapshot(s) from t≈%.3f", baseline, amax, amax / baseline,
                    len(u_keep), t_on)
        return u_keep, P_keep

    last_t = u_ts[-1]
    F_ref = np.eye(cfg.gdim) + last_t * (target_F - np.eye(cfg.gdim))
    logger.info("  no buckling (||u_fluc|| stayed ~%.3g); LBA fallback at t=%.3f",
                baseline, last_t)
    return _lba_modes(cfg, F_ref, f"path_{tag}"), []


def _run_lba(cfg: _RunConfig, target_F: np.ndarray,
             tag: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Step as close to the bifurcation as possible (perturbation disabled — the
    solver rejects + halves dt at the instability) and do an LBA there. Returns
    u-space buckling modes only (no equilibrium stress field)."""
    solver = _build_solver(cfg, os.path.join(cfg.work_dir, f"path_{tag}"),
                           check_stability=True, perturb_post_buckling=False)
    try:
        solver(target_F)
        logger.info("  no instability encountered up to F̄_target")
    except RVEConvergenceError as exc:
        logger.info("  approached the bifurcation (%s)", exc)
    F_crit = np.asarray(solver._last_converged_Fbar)
    logger.info("  LBA at near-critical F̄ =\n%s", np.array2string(F_crit, precision=4))
    # The approach solve leaves F_bar at the rejected post-critical trial while u was
    # reset to the last converged step — restore the consistent (F_crit, u) state and
    # do the LBA in place, so we don't re-ramp a fresh solver back to F_crit.
    solver.F_bar.value[:] = solver._last_converged_Fbar
    solver.u.x.array[:] = solver._u_last.x.array
    solver.u.x.scatter_forward()
    return _lba_modes(cfg, F_crit, f"path_{tag}", solver=solver), []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(output_dir: str, V, result: EnrichmentModes,
                   pool_P_coords_src: "str | None") -> None:
    phi_dir = os.path.join(output_dir, "phi")
    if os.path.isdir(phi_dir):
        shutil.rmtree(phi_dir)
    os.makedirs(phi_dir, exist_ok=True)
    np.save(os.path.join(phi_dir, "phi_dof_coords.npy"), result.dof_coords)
    np.save(os.path.join(phi_dir, "singular_values.npy"), result.singular_values)
    for i in range(result.n_modes):
        np.save(os.path.join(phi_dir, f"phi_{i:03d}.npy"), result.phi[:, i].copy())

    # ParaView .bp (one timestep per mode) + singular-value plot.
    phi_fn = fem.Function(V, name="phi")
    writer = io.VTXWriter(V.mesh.comm, os.path.join(output_dir, "phi_modes.bp"),
                          [phi_fn], engine="BP4")
    try:
        for i in range(result.n_modes):
            phi_fn.x.array[:] = result.phi[:, i]
            phi_fn.x.scatter_forward()
            writer.write(float(i))
    finally:
        writer.close()
    _plot_singular_values(result.singular_values, result.n_modes,
                          os.path.join(output_dir, "singular_values.png"))

    # Pooled post-buckling snapshots (g=0 training data for the micromorphic ROM).
    pool_dir = os.path.join(output_dir, "snapshots_pool")
    if os.path.isdir(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir, exist_ok=True)
    np.save(os.path.join(pool_dir, "u_fluc.npy"), result.snapshots_u)
    np.save(os.path.join(pool_dir, "u_fluc_dof_coords.npy"), result.dof_coords)
    if result.snapshots_P.size:
        np.save(os.path.join(pool_dir, "P.npy"), result.snapshots_P)
        if pool_P_coords_src and os.path.exists(pool_P_coords_src):
            shutil.copy(pool_P_coords_src, os.path.join(pool_dir, "P_dof_coords.npy"))
    logger.info("Wrote %d φ mode(s) to %s (+ phi_modes.bp, singular_values.png, "
                "snapshots_pool/)", result.n_modes, phi_dir)


def _plot_singular_values(sv: np.ndarray, n_selected: int, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, sv.size + 1), sv / sv[0], "o-", ms=4)
    ax.axvline(n_selected + 0.5, color="r", ls="--", label=f"N = {n_selected}")
    ax.set(xlabel="mode index", ylabel=r"$\sigma_i/\sigma_0$",
           title="POD singular-value spectrum of buckling fields")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_buckling_modes(
    mesh_path: str, comm, gdim: int, material, *,
    degree: int = 2,
    quadrature_degree: "int | None" = None,
    lattice_vectors: "np.ndarray | None" = None,
    end_deformations: "list[np.ndarray] | None" = None,
    max_strain: float = 0.15,
    strategy: str = "traverse",
    gap_ratio: float = 2.0,
    pert_amplitude: float = 0.1,
    postbuckle_frac: float = 0.1,
    buckle_amp_ratio: float = 5.0,
    fallback_n_modes: int = 6,
    dt_init: "float | None" = None,
    dt_min: float = 1e-3,
    dt_max: "float | None" = None,
    good_newton_steps: int = 7,
    newton_options: "dict | None" = None,
    stability_options: "dict | None" = None,
    inner_product: str = "H1",
    output_dir: "str | None" = None,
) -> EnrichmentModes:
    """Find micromorphic enrichment modes ``φ`` from full-order buckling of an RVE.

    Parameters
    ----------
    mesh_path, comm, gdim, material : the RVE (as for ``ch1.MicroSolver``).
    degree : Lagrange degree of the displacement space.
    quadrature_degree : integration degree of the inner CH1 solves (``None`` →
        DOLFINx automatic). Set it to match the training / online rule.
    lattice_vectors : ``None`` → axis-aligned box periodicity; ``(gdim, gdim)``
        array (2D) → arbitrary polygon periodicity.
    end_deformations : list of target ``F̄`` to drive (each ramped from ``I``).
        Defaults to a single equal-biaxial/triaxial compression of ``max_strain``.
    max_strain : compressive strain for the default symmetric load case.
    strategy : ``"traverse"`` (perturb past the bifurcation, keep post-buckling
        snapshots + P; LBA fallback) or ``"lba"`` (don't traverse — approach the
        bifurcation and do an LBA there).
    gap_ratio : POD singular-value drop / eigenvalue jump used to pick the mode
        counts (the main modeling knob).
    pert_amplitude, postbuckle_frac, buckle_amp_ratio, fallback_n_modes :
        perturbation kick, developed-pattern fraction, buckle-detection ratio,
        and number of LBA eigenpairs (see module docstring).
    dt_init, dt_min, dt_max, good_newton_steps, newton_options, stability_options :
        time-stepper / Newton / stability options for the inner CH1 solves.
    inner_product : POD inner product for the modes (``"H1"`` or ``"L2"``).
    output_dir : if given, write ``phi/``, ``phi_modes.bp``,
        ``singular_values.png`` and ``snapshots_pool/``.

    Returns
    -------
    EnrichmentModes with the ``(n_dofs, N)`` ``phi`` basis and diagnostics.
    """
    if strategy not in ("traverse", "lba"):
        raise ValueError(f"strategy must be 'traverse' or 'lba', got {strategy!r}")
    if end_deformations is None:
        end_deformations = make_symmetric_compression(max_strain, gdim)
    # A small step is used for both strategies so "lba" steps *through* the
    # bifurcation and the stability check catches it. The check only sees the
    # ``nev`` eigenvalues closest to zero, so once a step lands well past the
    # bifurcation the (now far-from-zero) negative mode can drop out of that set
    # and be missed ("tunnelling"); a small dt keeps the solve inside the
    # detection window. (0.1 already tunnels for the square RVE; 0.05 is safe.)
    if dt_init is None:
        dt_init = 0.05
    if dt_max is None:
        dt_max = dt_init

    work_dir = output_dir or tempfile.mkdtemp(prefix="mm_mode_extract_")
    os.makedirs(work_dir, exist_ok=True)
    cfg = _RunConfig(
        mesh_path=mesh_path, comm=comm, gdim=gdim, material=material, degree=degree,
        quadrature_degree=quadrature_degree,
        lattice_vectors=lattice_vectors, work_dir=work_dir,
        gap_ratio=gap_ratio, pert_amplitude=pert_amplitude,
        postbuckle_frac=postbuckle_frac, buckle_amp_ratio=buckle_amp_ratio,
        fallback_n_modes=fallback_n_modes,
        newton_options=newton_options or _DEFAULT_NEWTON,
        stability_options=stability_options or _DEFAULT_STABILITY,
        timestepper_options={"t_end": 1.0, "dt_init": dt_init, "dt_min": dt_min,
                             "dt_max": dt_max, "good_newton_steps": good_newton_steps},
    )
    runner = _run_traverse if strategy == "traverse" else _run_lba

    # 1-2. Run each end deformation; pool the buckling fields.
    pool_u: list[np.ndarray] = []
    pool_P: list[np.ndarray] = []
    for i, F in enumerate(end_deformations):
        F = np.asarray(F, dtype=float)
        logger.info("── Deformation %d/%d (strategy=%s): F̄_target =\n%s",
                    i + 1, len(end_deformations), strategy,
                    np.array2string(F, precision=4))
        u, P = runner(cfg, F, f"def{i}")
        pool_u.extend(u)
        pool_P.extend(P)
    if not pool_u:
        raise RuntimeError("No buckling fields collected — check max_strain / load "
                           "cases (did anything buckle?).")

    # 3. H¹-POD → φ basis, mode count from the first singular-value drop.
    with silence_c_stdout():
        mesh = io.gmsh.read_from_msh(mesh_path, comm, 0, gdim=gdim).mesh
    V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    pod = POD(np.asarray(pool_u), V, inner_product=inner_product)
    sv = np.sqrt(np.maximum(pod.eigenvalues, 0.0))
    N = _n_modes_by_gap(sv, gap_ratio)
    logger.info("Pooled %d field(s) (%d with P); singular values σ_i/σ_0 = %s",
                len(pool_u), len(pool_P), np.array2string(sv[:12] / sv[0], precision=3))
    logger.info("→ N = %d modes (first σ-drop ≥ %.1f×; energy-99.99%% would keep %d)",
                N, gap_ratio, pod.n_modes(0.9999))

    result = EnrichmentModes(
        phi=pod.basis[:, :N].copy(), singular_values=sv, n_modes=N,
        dof_coords=V.tabulate_dof_coordinates()[:, :gdim].copy(),
        snapshots_u=np.asarray(pool_u),
        snapshots_P=np.asarray(pool_P) if pool_P else np.empty((0, 0)),
    )

    if output_dir is not None:
        # locate a P dof-coords file (only present for traversed paths)
        P_coords_src = None
        for c in glob(os.path.join(work_dir, "path_*", "snapshots", "P_dof_coords.npy")):
            P_coords_src = c
            break
        _write_outputs(output_dir, V, result, P_coords_src)
    return result


# ---------------------------------------------------------------------------
# Analytical enrichment modes (φ supplied as closed-form expressions)
# ---------------------------------------------------------------------------

def _normalize_mode(phi: fem.Function, dx, comm, normalize: "str | None",
                    rve_volume: "float | None") -> None:
    """Scale ``phi`` in place. ``normalize``: "mean" → (1/|Q|) ∫ ‖φ‖ dX = 1;
    "h1" → unit H¹ norm; None → unchanged."""
    if normalize is None:
        return
    if normalize == "mean":
        integ = comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.sqrt(ufl.dot(phi, phi)) * dx)), op=MPI.SUM)
        if rve_volume is None:
            rve_volume = comm.allreduce(
                fem.assemble_scalar(fem.form(fem.Constant(phi.function_space.mesh, 1.0) * dx)),
                op=MPI.SUM)
        C = integ / rve_volume
    elif normalize == "h1":
        h1_sq = comm.allreduce(fem.assemble_scalar(fem.form(
            (ufl.inner(phi, phi) + ufl.inner(ufl.grad(phi), ufl.grad(phi))) * dx)), op=MPI.SUM)
        C = np.sqrt(h1_sq)
    else:
        raise ValueError(f"normalize must be 'mean', 'h1', or None, got {normalize!r}")
    if C > 0.0:
        phi.x.array[:] /= C
        phi.x.scatter_forward()


def set_analytical_modes(phi_functions, mode_functions, *, dx, comm,
                         normalize: "str | None" = "mean",
                         rve_volume: "float | None" = None) -> None:
    """Fill each :class:`dolfinx.fem.Function` in ``phi_functions`` **in place**
    from the matching callable in ``mode_functions``, then normalise.

    ``mode_functions`` is a callable, or a list of callables, each mapping a
    coordinate array of shape ``(n_points, gdim)`` to mode values of shape
    ``(n_points, gdim)`` — the analytical patterning field for one mode (e.g. a
    sinusoid). Each must be **cell-periodic**. ``dx`` and ``comm`` are the RVE
    measure and communicator (for the normalisation integrals); see
    ``_normalize_mode`` for ``normalize`` / ``rve_volume``.

    This is the in-place core used by :func:`make_analytical_modes`, and the
    handle for populating an existing solver's ``self._phi`` (call
    ``rebuild_constraints()`` afterwards)."""
    if callable(mode_functions):
        mode_functions = [mode_functions]
    if len(mode_functions) != len(phi_functions):
        raise ValueError(f"{len(mode_functions)} mode function(s) for "
                         f"{len(phi_functions)} φ slot(s)")
    for phi, fn in zip(phi_functions, mode_functions):
        V = phi.function_space
        gdim = V.mesh.geometry.dim
        bs = V.dofmap.bs
        coords = V.tabulate_dof_coordinates()[:, :gdim]
        vals = np.asarray(fn(coords), dtype=np.float64)
        if vals.shape != (coords.shape[0], gdim):
            raise ValueError(f"mode function returned shape {vals.shape}, expected "
                             f"{(coords.shape[0], gdim)} = (n_nodes, gdim)")
        for d in range(gdim):
            phi.x.array[d::bs] = vals[:, d]
        phi.x.scatter_forward()
        _normalize_mode(phi, dx, comm, normalize, rve_volume)


def _write_phi_modes(output_dir: str, V, result: EnrichmentModes) -> None:
    """Write ``phi/phi_<i>.npy`` + dof coords and a ``phi_modes.bp`` (no POD /
    snapshot diagnostics — used by the analytical generator)."""
    phi_dir = os.path.join(output_dir, "phi")
    if os.path.isdir(phi_dir):
        shutil.rmtree(phi_dir)
    os.makedirs(phi_dir, exist_ok=True)
    np.save(os.path.join(phi_dir, "phi_dof_coords.npy"), result.dof_coords)
    for i in range(result.n_modes):
        np.save(os.path.join(phi_dir, f"phi_{i:03d}.npy"), result.phi[:, i].copy())
    phi_fn = fem.Function(V, name="phi")
    writer = io.VTXWriter(V.mesh.comm, os.path.join(output_dir, "phi_modes.bp"),
                          [phi_fn], engine="BP4")
    try:
        for i in range(result.n_modes):
            phi_fn.x.array[:] = result.phi[:, i]
            phi_fn.x.scatter_forward()
            writer.write(float(i))
    finally:
        writer.close()
    logger.info("Wrote %d φ mode(s) to %s (+ phi_modes.bp)", result.n_modes, phi_dir)


def make_analytical_modes(
    mesh_path: str, comm, gdim: int, mode_functions, *,
    degree: int = 2,
    normalize: "str | None" = "mean",
    rve_volume: "float | None" = None,
    output_dir: "str | None" = None,
) -> EnrichmentModes:
    """Generate micromorphic enrichment modes ``φ`` from analytical expressions.

    Unlike :func:`extract_buckling_modes` (which learns φ from full-order RVE
    buckling), this builds φ directly from closed-form patterning fields you
    supply as callables — e.g. a sinusoidal pattern (van Bree et al. Eq. 56).

    Parameters
    ----------
    mesh_path, comm, gdim : the RVE (as for ``ch1.MicroSolver``).
    mode_functions : a callable, or list of callables, each mapping a coordinate
        array ``(n_points, gdim)`` → mode values ``(n_points, gdim)``. The number
        of callables sets ``N``. Each must be cell-periodic.
    degree : Lagrange degree of the displacement space (must match the solver the
        modes will be loaded into).
    normalize : "mean" (⟨‖φ‖⟩_Q = 1), "h1" (unit H¹), or None.
    rve_volume : cell volume |Q| for "mean" normalisation; if None, ∫ 1 dx.
    output_dir : if given, write ``phi/`` + ``phi_modes.bp``.

    Returns an :class:`EnrichmentModes` (same type as ``extract_buckling_modes``),
    so it plugs into ``mm.MicroSolver.load_buckling_modes`` identically.
    """
    if callable(mode_functions):
        mode_functions = [mode_functions]
    with silence_c_stdout():
        mesh = io.gmsh.read_from_msh(mesh_path, comm, 0, gdim=gdim).mesh
    V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
    dx = ufl.Measure("dx", domain=mesh)
    phis = [fem.Function(V, name=f"phi_{i}") for i in range(len(mode_functions))]
    set_analytical_modes(phis, mode_functions, dx=dx, comm=comm,
                         normalize=normalize, rve_volume=rve_volume)
    result = EnrichmentModes(
        phi=np.column_stack([p.x.array.copy() for p in phis]),
        singular_values=np.empty(0), n_modes=len(phis),
        dof_coords=V.tabulate_dof_coordinates()[:, :gdim].copy(),
        snapshots_u=np.empty((0, 0)), snapshots_P=np.empty((0, 0)),
    )
    logger.info("Generated %d analytical φ mode(s) (normalize=%s)",
                result.n_modes, normalize)
    if output_dir is not None:
        _write_phi_modes(output_dir, V, result)
    return result
