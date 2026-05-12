import numpy as np


class TimeStepper:
    """Adaptive time stepper with dt increase on fast Newton convergence and
    dt halving on Newton failure."""

    def __init__(self, t_end: float = 1.0, dt_init: float = 0.1,
                 dt_min: float = 1e-5, dt_max: float = 0.1,
                 good_newton_steps: int = 7):
        self.t_end = t_end
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self._good_newton_steps = good_newton_steps
        self.t_current = 0.0
        self.dt = dt_init
        self._trial_time: float = 0.0

    def step_forward(self) -> float:
        """Compute and cache the trial time for the next step."""
        self._trial_time = np.round(self.t_current + self.dt, 5)
        return self._trial_time

    def accept(self, n_newton_iters: int) -> None:
        """Advance t_current and optionally increase dt."""
        self.t_current = self._trial_time
        if n_newton_iters <= self._good_newton_steps:
            self.dt = min(self.dt * 1.5, self.dt_max)
        self.dt = min(self.dt, self.t_end - self.t_current)

    def reject(self) -> bool:
        """Halve dt. Returns True if simulation can continue, False if dt < dt_min."""
        self.dt /= 2
        return self.dt >= self.dt_min

    @property
    def finished(self) -> bool:
        return self.t_current >= self.t_end

    def reset(self) -> None:
        """Reset time stepper to initial state."""
        self.t_current = 0.0
        self.dt = self.dt_init