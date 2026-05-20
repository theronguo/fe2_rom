"""Finite-difference verification of the micromorphic effective tangent blocks.

For each macro variable μ ∈ {F̄, v, g} and each output quantity
Q ∈ {P̄, Π, Λ}, compare the analytical ``dQ/dμ`` block (returned by
``MicromorphicHyperelasticHomogenizationSolver`` via the ``TangentBlock``
adjoint machinery) against a central-difference of the corresponding output
at perturbed macro inputs.

Each ``solver(F̄, v, g)`` call ramps from the committed restart state (which
we never advance via ``commit()``), so perturbed solves are independent —
no path dependence between them.

Run:
    conda activate fe2_rom_env
    python verify_fd.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import numpy as np
from mpi4py import MPI

from fe2_rom.hyperelastic_solver import (
    MicromorphicHyperelasticHomogenizationSolver,
    NeoHookean,
    setup_logging,
)


# --- Setup ------------------------------------------------------------------
comm = MPI.COMM_WORLD
setup_logging(comm, level=logging.ERROR)  # FD loop is noisy; silence inner solves

HERE = os.path.dirname(__file__)
RVE_MESH = os.path.abspath(os.path.join(HERE, "rve.msh"))

E_micro, nu_micro = 3000.0, 0.30
mu_micro = E_micro / (2.0 * (1.0 + nu_micro))
lam_micro = E_micro * nu_micro / ((1.0 + nu_micro) * (1.0 - 2.0 * nu_micro))

N_MODES = 1
solver = MicromorphicHyperelasticHomogenizationSolver(
    mesh_path=RVE_MESH,
    comm=comm,
    gdim=2,
    material=NeoHookean(mu=mu_micro, lmbda=lam_micro),
    N=N_MODES,
    degree=2,
    check_stability=False,
    visualize_fields=[],
    # Tight Newton so FD noise floor is set by ε, not by residual tolerance.
    newton_options={"rel_tol": 1e-8, "abs_tol": 1e-12,
                    "max_iter": 50, "div_rel_tol": 10,
                    "switch_to_minres": True},
    timestepper_options={"t_end": 1.0, "dt_init": 1.0, "dt_min": 1e-5,
                         "dt_max": 1.0, "good_newton_steps": 5},
)

eigvals = solver.compute_linear_buckling_modes(N_MODES, visualize_modes=True)

# --- Baseline state ---------------------------------------------------------
# Mildly loaded: stays well below buckling so Newton converges in one step.
F0 = np.array([[0.95, 0.02],
               [-0.03,  1.03]])
v0 = np.array([0.8])
g0 = np.array([[-0.15, 0.08]])

gdim = 2
N = N_MODES
EPS = 1e-6


def evaluate(F, v, g):
    res = solver(F, v, g)
    return res[-1]


# Snapshot all baseline analytical tangents and outputs.
base = evaluate(F0, v0, g0)
analytical = {k: np.asarray(base[k]) for k in [
    "dPbar_dFbar", "dPbar_dv", "dPbar_dg",
    "dPi_dFbar",   "dPi_dv",   "dPi_dg",
    "dLambda_dFbar", "dLambda_dv", "dLambda_dg",
]}

# --- Helpers to perturb one scalar component at a time ----------------------

def _fd(plus, minus, key):
    return (np.asarray(plus[key]) - np.asarray(minus[key])) / (2.0 * EPS)


# d./dF̄ : sweep (k, l) ∈ gdim × gdim
fd_dPbar_dFbar = np.zeros((gdim, gdim, gdim, gdim))
fd_dPi_dFbar = np.zeros((N, gdim, gdim))
fd_dLambda_dFbar = np.zeros((N, gdim, gdim, gdim))
for k in range(gdim):
    for l in range(gdim):
        Fp = F0.copy(); Fp[k, l] += EPS
        Fm = F0.copy(); Fm[k, l] -= EPS
        plus = evaluate(Fp, v0, g0)
        minus = evaluate(Fm, v0, g0)
        fd_dPbar_dFbar[:, :, k, l]    = _fd(plus, minus, "Pbar")
        fd_dPi_dFbar[:, k, l]         = _fd(plus, minus, "Pi")
        fd_dLambda_dFbar[:, :, k, l]  = _fd(plus, minus, "Lambda")

# d./dv : sweep i ∈ N
fd_dPbar_dv = np.zeros((gdim, gdim, N))
fd_dPi_dv = np.zeros((N, N))
fd_dLambda_dv = np.zeros((N, gdim, N))
for i in range(N):
    vp = v0.copy(); vp[i] += EPS
    vm = v0.copy(); vm[i] -= EPS
    plus = evaluate(F0, vp, g0)
    minus = evaluate(F0, vm, g0)
    fd_dPbar_dv[:, :, i]   = _fd(plus, minus, "Pbar")
    fd_dPi_dv[:, i]        = _fd(plus, minus, "Pi")
    fd_dLambda_dv[:, :, i] = _fd(plus, minus, "Lambda")

# d./dg : sweep (i, d) ∈ N × gdim
fd_dPbar_dg = np.zeros((gdim, gdim, N, gdim))
fd_dPi_dg = np.zeros((N, N, gdim))
fd_dLambda_dg = np.zeros((N, gdim, N, gdim))
for i in range(N):
    for d in range(gdim):
        gp = g0.copy(); gp[i, d] += EPS
        gm = g0.copy(); gm[i, d] -= EPS
        plus = evaluate(F0, v0, gp)
        minus = evaluate(F0, v0, gm)
        fd_dPbar_dg[:, :, i, d]    = _fd(plus, minus, "Pbar")
        fd_dPi_dg[:, i, d]         = _fd(plus, minus, "Pi")
        fd_dLambda_dg[:, :, i, d]  = _fd(plus, minus, "Lambda")

fd = {
    "dPbar_dFbar": fd_dPbar_dFbar,
    "dPbar_dv":    fd_dPbar_dv,
    "dPbar_dg":    fd_dPbar_dg,
    "dPi_dFbar":   fd_dPi_dFbar,
    "dPi_dv":      fd_dPi_dv,
    "dPi_dg":      fd_dPi_dg,
    "dLambda_dFbar": fd_dLambda_dFbar,
    "dLambda_dv":    fd_dLambda_dv,
    "dLambda_dg":    fd_dLambda_dg,
}


# --- Report -----------------------------------------------------------------
ASSERT_BLOCKS = ["dPbar_dFbar", "dPbar_dv", "dPbar_dg",
                 "dPi_dFbar", "dPi_dv", "dPi_dg",
                 "dLambda_dFbar", "dLambda_dv", "dLambda_dg"]  # assert these blocks are correct within tolerance
TOL = 5e-3  # generous for FD with ε=1e-5 on widely-scaled blocks

if comm.rank == 0:
    print(f"ε = {EPS:g}, baseline F0={F0.flatten()} v0={v0} g0={g0.flatten()}")
    print()
    print(f"{'block':18s} {'|A|':>12s} {'|FD|':>12s} "
          f"{'|A−FD|':>12s} {'rel':>10s}  status")
    print("-" * 90)
    max_rel = 0.0
    fail = False
    for k in analytical:
        A = analytical[k]
        F_ = fd[k]
        nA = np.linalg.norm(A)
        nF = np.linalg.norm(F_)
        nD = np.linalg.norm(A - F_)
        rel = nD / max(nA, 1e-30)
        if rel < TOL:
            status = "OK   "
            max_rel = max(max_rel, rel)
        else:
            status = "FAIL "
            fail = True
        print(f"{k:18s} {nA:12.4e} {nF:12.4e} {nD:12.4e} {rel:10.2e}  {status}")
    print("-" * 90)
    print(f"max relative error over asserted blocks {ASSERT_BLOCKS}: {max_rel:.3e}")
    if fail:
        raise SystemExit("FD verification: at least one asserted block exceeded tolerance")
