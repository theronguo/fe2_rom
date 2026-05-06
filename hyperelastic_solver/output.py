import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import io

logger = logging.getLogger(__name__)


class VTXManager:
    """Thin wrapper around dolfinx VTXWriter for BP4 time-series output.

    write() and close() are collective — call on all MPI ranks.
    """

    def __init__(self, comm, path: str, fields: list):
        self._writer = io.VTXWriter(comm, path, fields, engine="BP4")

    def write(self, t: float) -> None:
        self._writer.write(t)

    def close(self) -> None:
        self._writer.close()


class ReactionForceLogger:
    """Accumulates applied-displacement / reaction-force pairs and saves to PNG + CSV.

    record() is called on all ranks but only uses the values.
    save() writes files on rank 0 only.
    """

    def __init__(self):
        self.displacements: list[float] = []
        self.forces: list[float] = []

    def record(self, disp: float, force: float) -> None:
        self.displacements.append(disp)
        self.forces.append(force)

    def save(self, comm, png_path: str, csv_path: str) -> None:
        if comm.rank == 0 and len(self.displacements) > 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(self.displacements, self.forces, "b-o", markersize=4, linewidth=1.5)
            ax.set_xlabel("Applied displacement (m)")
            ax.set_ylabel("Reaction force z (N)")
            ax.set_title("Reaction force vs applied displacement")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            np.savetxt(
                csv_path,
                np.column_stack([self.displacements, self.forces]),
                delimiter=",",
                header="applied_displacement,reaction_force_z",
                comments="",
            )
            logger.info("Saved %s and %s", png_path, csv_path)
