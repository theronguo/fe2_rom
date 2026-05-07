"""Convert an STL surface mesh to a Gmsh-compatible tetrahedral .msh file.

Pipeline:
  1. pymeshlab  — clean and remesh the STL surface
  2. tetgen     — tetrahedralise the closed surface
  3. meshio     — write a Gmsh 2.2 .msh with physical/geometrical tags
                  that dolfinx can read via io.gmshio.read_from_msh

Usage:
    python stl_to_msh.py                      # uses defaults below
    from stl_to_msh import stl_to_msh
    stl_to_msh("my_part.stl", "my_part.msh")
"""

import logging
import os
import pathlib

import meshio
import numpy as np
import pymeshlab
import pyvista as pv
import tetgen

logger = logging.getLogger(__name__)


def stl_to_msh(
    stl_path: str,
    msh_path: str,
    *,
    remesh_iterations: int = 30,
    remesh_target_pct: float = 1.1,
    close_holes_maxsize: int = 1000,
    merge_threshold_pct: float = 0.001,
    tet_mindihedral: float = 10.0,
    tet_minratio: float = 2.0,
) -> str:
    """Clean, remesh, and tetrahedralise an STL file, writing a .msh.

    Parameters
    ----------
    stl_path:
        Input surface mesh (STL format).
    msh_path:
        Output tetrahedral mesh (Gmsh 2.2 ASCII format).
    remesh_iterations:
        Isotropic remeshing iterations (more → more uniform triangles).
    remesh_target_pct:
        Target edge length as a percentage of the bounding-box diagonal.
    close_holes_maxsize:
        Maximum hole size (in edges) that will be filled.
    merge_threshold_pct:
        Vertex merge threshold as a percentage of bounding-box diagonal.
    tet_mindihedral:
        Minimum dihedral angle constraint for TetGen (degrees, lower = more relaxed).
    tet_minratio:
        Maximum radius/edge ratio for TetGen (higher = more relaxed).

    Returns
    -------
    str
        Absolute path to the written .msh file.
    """
    stl_path = pathlib.Path(stl_path).resolve()
    msh_path = pathlib.Path(msh_path).resolve()

    # ------------------------------------------------------------------ #
    # Step 1: Surface cleanup                                              #
    # ------------------------------------------------------------------ #
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(stl_path))
    logger.info("Loaded '%s'  initial faces: %d", stl_path.name, ms.current_mesh().face_number())

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_merge_close_vertices(
        threshold=pymeshlab.PercentageValue(merge_threshold_pct)
    )
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_close_holes(maxholesize=close_holes_maxsize)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_null_faces()

    clean_stl = stl_path.with_suffix(".clean.stl")
    ms.save_current_mesh(str(clean_stl))
    logger.info("After cleanup:   faces: %d", ms.current_mesh().face_number())

    # ------------------------------------------------------------------ #
    # Step 2: Isotropic remeshing                                         #
    # ------------------------------------------------------------------ #
    ms2 = pymeshlab.MeshSet()
    ms2.load_new_mesh(str(clean_stl))
    ms2.meshing_isotropic_explicit_remeshing(
        iterations=remesh_iterations,
        targetlen=pymeshlab.PercentageValue(remesh_target_pct),
    )
    remeshed_stl = stl_path.with_suffix(".remeshed.stl")
    ms2.save_current_mesh(str(remeshed_stl))
    logger.info("After remeshing: faces: %d", ms2.current_mesh().face_number())

    # Quick manifold check
    surface = pv.read(str(remeshed_stl))
    logger.info("Watertight: %s  Open edges: %d", surface.is_manifold, surface.n_open_edges)

    # ------------------------------------------------------------------ #
    # Step 3: Tetrahedralisation                                          #
    # ------------------------------------------------------------------ #
    tet_gen = tetgen.TetGen(surface)
    tet_gen.make_manifold()
    tet_gen.tetrahedralize(
        order=1,
        mindihedral=tet_mindihedral,
        minratio=tet_minratio,
        nobisect=True,
    )
    vtu_path = stl_path.with_suffix(".vtu")
    tet_gen.grid.save(str(vtu_path))
    logger.info("Tetrahedralised: %d cells", tet_gen.grid.n_cells)

    # ------------------------------------------------------------------ #
    # Step 4: Add Gmsh physical/geometrical tags and write .msh           #
    # ------------------------------------------------------------------ #
    vol_mesh = meshio.read(str(vtu_path))

    physical_data = []
    geometrical_data = []
    for block in vol_mesh.cells:
        n = len(block.data)
        if block.type == "tetra":
            physical_data.append(np.full(n, 1, dtype=np.int32))
            geometrical_data.append(np.full(n, 1, dtype=np.int32))
        elif block.type == "triangle":
            physical_data.append(np.full(n, 2, dtype=np.int32))
            geometrical_data.append(np.full(n, 2, dtype=np.int32))
        else:
            physical_data.append(np.full(n, 99, dtype=np.int32))
            geometrical_data.append(np.full(n, 99, dtype=np.int32))

    vol_mesh.cell_data = {
        "gmsh:physical": physical_data,
        "gmsh:geometrical": geometrical_data,
    }
    vol_mesh.field_data = {
        "volume": np.array([1, 3], dtype=np.int32),
        "boundary": np.array([2, 2], dtype=np.int32),
    }

    meshio.write(str(msh_path), vol_mesh, file_format="gmsh22", binary=False)
    logger.info("Written '%s'", msh_path)

    os.remove(clean_stl)
    os.remove(remeshed_stl)
    os.remove(vtu_path)
    return str(msh_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    stl_to_msh("stl_files/meta_circ.stl", "model.msh")
