import contextlib
import logging
import os
import sys
import warnings


class _Rank0Filter(logging.Filter):
    """Suppress log records emitted by non-root MPI ranks."""

    def __init__(self, comm):
        super().__init__()
        self._comm = comm

    def filter(self, record):
        return self._comm.rank == 0


def setup_logging(comm, level: int = logging.INFO) -> None:
    """Configure package-wide logging for an MPI run.

    Only rank-0 emits records. Call once before constructing the solver.

    Parameters
    ----------
    comm:
        MPI communicator (e.g. MPI.COMM_WORLD).
    level:
        Root log level. Use logging.DEBUG to see per-Newton-iteration residuals.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_Rank0Filter(comm))
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler], force=True)

    # Suppress spurious mpi4py struct-size mismatch warning that originates
    # from a binary layout difference between the mpi4py wheel and the system
    # MPI — it does not affect correctness.
    warnings.filterwarnings(
        "ignore",
        message="mpi4py.MPI.Session size changed",
        category=RuntimeWarning,
    )


@contextlib.contextmanager
def silence_c_stdout():
    """Silence file-descriptor-1 output for the duration of the context.

    Python's ``contextlib.redirect_stdout`` only intercepts ``sys.stdout``
    writes — it doesn't touch the underlying file descriptor.  Native code
    (gmsh, MUMPS, ParMETIS, …) writes directly to fd 1 and bypasses Python,
    so we redirect fd 1 itself to /dev/null instead.
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(devnull)
        os.close(saved_fd)
