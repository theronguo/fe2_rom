"""Differentiable right polar decomposition F = R U in JAX.

``U = sqrt(FᵀF)`` is computed with a coupled Newton–Schulz iteration on the
Frobenius-normalised C = FᵀF. Unlike ``jnp.linalg.eigh`` this is smooth and
batch/vmap-friendly with no degenerate-eigenvalue gradient issues at U ≈ I
(the reference state of every simulation), and it composes cleanly with
``jax.grad`` / ``jax.jacfwd`` first- and second-derivative transforms.

Convergence requires the eigenvalues of C/‖C‖_F to lie in (0, 2), which holds
for any SPD C; the iteration is quadratically convergent once contraction sets
in. For the stretch ranges of interest (λ ∈ [0.5, 1.5]) the default iteration
count reaches float64 machine precision.
"""
from __future__ import annotations

import jax.numpy as jnp

_DEFAULT_ITERS = 18


def _fro_norm(C):
    """Frobenius norm over the last two axes, keepdims for broadcasting."""
    return jnp.sqrt(jnp.sum(C ** 2, axis=(-2, -1), keepdims=True))


def right_stretch(F, iters: int = _DEFAULT_ITERS):
    """Right stretch tensor U = sqrt(FᵀF) for F of shape (..., d, d)."""
    C = jnp.swapaxes(F, -1, -2) @ F
    s = _fro_norm(C)
    A = C / s
    d = F.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(d, dtype=F.dtype), A.shape)
    Y, Z = A, eye
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * jnp.sqrt(s)


def polar(F, iters: int = _DEFAULT_ITERS):
    """Right polar decomposition: returns (R, U) with F = R U.

    Uses the inverse-sqrt iterate (Z → ‖C‖_F^{1/2} C^{-1/2}), so no extra
    linear solve is needed for R = F C^{-1/2}.
    """
    C = jnp.swapaxes(F, -1, -2) @ F
    s = _fro_norm(C)
    A = C / s
    d = F.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(d, dtype=F.dtype), A.shape)
    Y, Z = A, eye
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    U = Y * jnp.sqrt(s)
    R = F @ (Z / jnp.sqrt(s))
    return R, U
