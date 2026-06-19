"""Differentiable macro-solve wrappers for fitting NN effective-energy weights
to macroscopic observations (e.g. a DNS reaction curve).

``MacroFit`` turns a built macro solver + its :class:`~fe2_rom.nn.model.EnergyNet`
into a callable ``fit(theta) -> reaction(schedule)`` with an analytic Jacobian
``fit.jac(theta) -> d reaction / d theta`` via a per-load-level discrete adjoint,
shaped for :func:`scipy.optimize.least_squares`.
"""
from fe2_rom.fit.macrofit import MacroFit

__all__ = ["MacroFit"]
