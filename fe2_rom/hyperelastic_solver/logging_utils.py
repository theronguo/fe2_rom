import contextlib
import contextvars
import logging
import os
import sys
import warnings


_current_qp: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "fe2_rom_current_qp", default=None,
)


@contextlib.contextmanager
def qp_context(qp: int):
    """Tag every log record emitted in this block with ``record.qp = qp``.

    Use from the macro driver around each inner RVE call so nested-solver logs
    can be traced back to their originating macro quadrature point.
    """
    token = _current_qp.set(qp)
    try:
        yield
    finally:
        _current_qp.reset(token)


class _RankAwareFilter(logging.Filter):
    """Let WARNING+ through on every rank; gate INFO/DEBUG to rank-0.

    Also stamps ``record.rank`` so the formatter can include it.
    """

    def __init__(self, comm):
        super().__init__()
        self._comm = comm

    def filter(self, record):
        record.rank = self._comm.rank
        record.qp = _current_qp.get()
        if record.levelno >= logging.WARNING:
            return True
        if getattr(record, "all_ranks", False):
            return True
        return self._comm.rank == 0


class _RankQPFormatter(logging.Formatter):
    """Formatter that inserts ``qp=<n>`` into the rank tag when set."""

    def format(self, record):
        s = super().format(record)
        qp = getattr(record, "qp", None)
        if qp is not None:
            s = s.replace(f"[r{record.rank}]", f"[r{record.rank} qp={qp}]", 1)
        return s


def setup_logging(comm, level: int = logging.INFO) -> None:
    """Configure package-wide logging for an MPI run.

    Rank-0 emits at the requested ``level``; non-root ranks emit WARNING and
    above only, so errors from any rank surface but INFO/DEBUG chatter stays
    single-stream. Every line is tagged with its emitting rank.

    Parameters
    ----------
    comm:
        MPI communicator (e.g. MPI.COMM_WORLD).
    level:
        Root log level. Use logging.DEBUG to see per-Newton-iteration residuals.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RankAwareFilter(comm))
    handler.setFormatter(_RankQPFormatter(
        fmt="%(asctime)s  %(levelname)-8s  [r%(rank)d]  %(name)-40s  %(message)s",
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


class _AllRanksStamp(logging.Filter):
    """Stamp every record passing through with ``all_ranks=True``.

    Attach to a logger to make its INFO/DEBUG records bypass the rank-0 gate
    installed by :func:`setup_logging`.
    """

    def filter(self, record):
        record.all_ranks = True
        return True


def broadcast_logger(*names: str, level: int = logging.INFO) -> None:
    """Make the given loggers emit on every rank (not just rank-0).

    Use this from a driver to surface per-rank INFO chatter from nested solvers
    when running under mpirun. Idempotent — safe to call repeatedly.
    """
    for name in names:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        if not any(isinstance(f, _AllRanksStamp) for f in lg.filters):
            lg.addFilter(_AllRanksStamp())


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
