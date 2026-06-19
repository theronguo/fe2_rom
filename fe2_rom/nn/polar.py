"""Differentiable right polar decomposition F = R U in torch.

``U = sqrt(FᵀF)`` is computed with a coupled Newton–Schulz iteration on the
Frobenius-normalised C = FᵀF. Unlike ``torch.linalg.eigh`` this is smooth and
batch/vmap-friendly with no degenerate-eigenvalue gradient issues at U ≈ I
(the reference state of every simulation), and it composes cleanly with
``torch.func`` first- and second-derivative transforms.

Convergence requires the eigenvalues of C/‖C‖_F to lie in (0, 2), which holds
for any SPD C; the iteration is quadratically convergent once contraction sets
in. For the stretch ranges of interest (λ ∈ [0.5, 1.5]) the default iteration
count reaches float64 machine precision.
"""
from __future__ import annotations

import torch

_DEFAULT_ITERS = 18


def right_stretch(F: torch.Tensor, iters: int = _DEFAULT_ITERS) -> torch.Tensor:
    """Right stretch tensor U = sqrt(FᵀF) for F of shape (..., d, d)."""
    C = F.mT @ F
    s = torch.linalg.matrix_norm(C, ord="fro", keepdim=True)
    A = C / s
    d = F.shape[-1]
    eye = torch.eye(d, dtype=F.dtype, device=F.device).expand_as(A)
    Y, Z = A, eye
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * torch.sqrt(s)


def polar(F: torch.Tensor, iters: int = _DEFAULT_ITERS):
    """Right polar decomposition: returns (R, U) with F = R U.

    Uses the inverse-sqrt iterate (Z → ‖C‖_F^{1/2} C^{-1/2}), so no extra
    linear solve is needed for R = F C^{-1/2}.
    """
    C = F.mT @ F
    s = torch.linalg.matrix_norm(C, ord="fro", keepdim=True)
    A = C / s
    d = F.shape[-1]
    eye = torch.eye(d, dtype=F.dtype, device=F.device).expand_as(A)
    Y, Z = A, eye
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    U = Y * torch.sqrt(s)
    R = F @ (Z / torch.sqrt(s))
    return R, U
