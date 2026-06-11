"""First-order homogenized effective stiffness of a periodic RVE.

A thin diagnostic wrapper around :class:`fe2_rom.ch1.MicroSolver`: build the RVE
on a mesh + material, drive it to a macroscopic deformation state ``F̄`` (default
the undeformed identity) and read back the homogenized first Piola stress ``P̄``
together with the first-order tangent ``A = dP̄/dF̄``.  Used for mesh-convergence
studies and quick sanity checks of an RVE's effective response.
"""
import logging

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from fe2_rom.ch1.microsolver import MicroSolver
from fe2_rom.hyperelastic_solver.material import MaterialModel

logger = logging.getLogger(__name__)


def effective_stiffness(
    mesh_path: str,
    material: MaterialModel,
    *,
    gdim: int = 3,
    comm=None,
    state: "np.ndarray | None" = None,
    rve_volume: "float | None" = None,
    degree: int = 1,
    lattice_vectors: "np.ndarray | None" = None,
    corner_periodic: bool = False,
    objective_reduction: bool = False,
    newton_options: "dict | None" = None,
    timestepper_options: "dict | None" = None,
    output_dir: str = "output_homog",
) -> dict:
    """Homogenized first-order tangent ``A = dP̄/dF̄`` of a periodic RVE.

    Builds a :class:`~fe2_rom.ch1.MicroSolver` on ``mesh_path`` with ``material``,
    drives it from the undeformed state to the macroscopic deformation gradient
    ``state`` (default ``F̄ = I``) and returns the converged homogenized
    quantities.

    Parameters
    ----------
    mesh_path : str
        Path to the periodic RVE ``.msh`` file.
    material : MaterialModel
        Constitutive law for the solid phase.
    gdim : int
        Geometric dimension (2 or 3).
    comm : MPI communicator, optional
        Defaults to ``MPI.COMM_WORLD``.
    state : (gdim, gdim) array, optional
        Macroscopic deformation gradient ``F̄`` to evaluate the tangent at.
        Defaults to the identity (undeformed reference).
    rve_volume : float, optional
        Exact cell volume ``|Q|`` for averaging; if ``None`` the mesh bounding
        box is used (correct only when the RVE fills its box).
    degree, lattice_vectors, corner_periodic, objective_reduction :
        Forwarded to :class:`~fe2_rom.ch1.MicroSolver`.

    Returns
    -------
    dict with keys
        ``"Fbar"``    : the applied F̄,
        ``"Pbar"``    : homogenized first Piola stress P̄,
        ``"A"``       : first-order tangent ``A[i,j,k,l] = dP̄_ij/dF̄_kl``,
        ``"relative_density"`` : solid volume / cell volume,
        ``"n_dofs"``, ``"n_cells"`` : mesh size descriptors.
    """
    comm = MPI.COMM_WORLD if comm is None else comm
    state = np.eye(gdim) if state is None else np.asarray(state, dtype=float)
    if state.shape != (gdim, gdim):
        raise ValueError(
            f"state must have shape ({gdim}, {gdim}); got {state.shape}.")

    if newton_options is None:
        newton_options = {"rel_tol": 1e-9, "abs_tol": 1e-9, "max_iter": 30}
    if timestepper_options is None:
        timestepper_options = {"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5,
                               "dt_max": 1.0, "good_newton_steps": 5}

    rve = MicroSolver(
        mesh_path=mesh_path, comm=comm, gdim=gdim, material=material,
        degree=degree, output_dir=output_dir,
        check_stability=False, visualize_fields=[],
        rve_volume=rve_volume, lattice_vectors=lattice_vectors,
        corner_periodic=corner_periodic, objective_reduction=objective_reduction,
        averages_only_final=True,
        newton_options=newton_options, timestepper_options=timestepper_options,
    )

    out = rve(state)[-1]
    A = np.asarray(out["dPbar_dFbar"], dtype=float)
    Pbar = np.asarray(out["Pbar"], dtype=float)

    cell_vol = (float(rve_volume) if rve_volume is not None
                else float(np.prod(rve.maxs - rve.mins)))
    solid_vol = comm.allreduce(
        fem.assemble_scalar(fem.form(1.0 * rve.dx)), op=MPI.SUM)
    n_cells = rve._mesh.topology.index_map(gdim).size_global
    n_dofs = rve.V.dofmap.index_map.size_global * rve.V.dofmap.index_map_bs

    return {
        "Fbar": np.asarray(out["Fbar"], dtype=float),
        "Pbar": Pbar,
        "A": A,
        "relative_density": solid_vol / cell_vol,
        "solid_volume": solid_vol,
        "cell_volume": cell_vol,
        "n_dofs": int(n_dofs),
        "n_cells": int(n_cells),
    }


def uniaxial_moduli(A: np.ndarray, axes=None) -> dict:
    """Uniaxial-strain and uniaxial-stress moduli from a first-order tangent.

    For each axis ``a`` (default all):
      * ``C_strain = A[a,a,a,a]`` — uniaxial *strain* modulus (lateral fixed);
      * ``E_stress`` — uniaxial *stress* modulus (lateral free), the Schur
        complement obtained by holding ``dF̄_aa`` and relaxing the other
        ``gdim²-1`` components so every ``P̄`` component except ``(a,a)`` vanishes.

    At a stress-free reference (e.g. ``F̄ = I``) the tangent ``A`` carries a
    rotational null space (``A : W = 0`` for skew ``W``), so the "free" block is
    rank-deficient.  The relaxation is therefore solved in the minimum-norm
    (least-squares) sense, which selects the rotation-free (symmetric) solution;
    the RHS is orthogonal to that null space, so the resulting ``E_stress`` is
    independent of the chosen solution and reduces to the ordinary Schur
    complement whenever the block is full rank (a deformed/stressed state).

    Returns ``{axis: (C_strain, E_stress)}``.
    """
    A = np.asarray(A, dtype=float)
    gdim = A.shape[0]
    n = gdim * gdim
    M = A.reshape(n, n)
    axes = range(gdim) if axes is None else axes
    res = {}
    for a in axes:
        aa = a * gdim + a
        free = [k for k in range(n) if k != aa]
        Mff = M[np.ix_(free, free)]
        Mfz = M[np.ix_(free, [aa])]
        dF_free = np.linalg.lstsq(Mff, -Mfz, rcond=1e-12)[0].ravel()
        E_stress = float(M[aa, aa] + M[aa, free] @ dF_free)
        res[a] = (float(M[aa, aa]), E_stress)
    return res
