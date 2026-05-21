"""Typed exceptions for solver-level failures.

These are caught by outer drivers (e.g. the FE² macro solver) to distinguish
*recoverable* solver bailouts — Newton/timestepper gave up — from genuine bugs.
"""


class RVEConvergenceError(RuntimeError):
    """Raised by an RVE solver when its time-stepper gives up (min dt reached).

    The macro solver catches this, allreduces a failure flag across ranks, and
    rejects the macro step collectively.
    """
