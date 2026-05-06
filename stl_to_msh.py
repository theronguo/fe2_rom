import pymeshlab

ms = pymeshlab.MeshSet()
ms.load_new_mesh("model.stl")

print("Initial faces:", ms.current_mesh().face_number())

# 1. Basic cleanup first
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_remove_null_faces()
ms.meshing_remove_unreferenced_vertices()

# 2. Merge near-coincident vertices (KEY for OpenSCAD output)
ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))

# 3. Remove non-manifold edges/vertices (this may open holes)
ms.meshing_repair_non_manifold_edges()      # default removes offending faces
ms.meshing_repair_non_manifold_vertices()

# 4. Now safe to close holes
ms.meshing_close_holes(maxholesize=1000)

# 5. Final cleanup
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_null_faces()

ms.save_current_mesh("model_clean.stl")
print("Final faces:", ms.current_mesh().face_number())


import pyvista as pv
mesh = pv.read("model_clean.stl")
print("Watertight:", mesh.is_manifold)
print("N open edges:", mesh.n_open_edges)


import pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh("model.stl")

# Isotropic explicit remeshing — fixes most pathologies
ms.meshing_isotropic_explicit_remeshing(
    iterations=6,
    targetlen=pymeshlab.PercentageValue(0.8),  # ~1% of bbox diagonal
)
ms.save_current_mesh("model_remeshed.stl")

import pyvista as pv
import tetgen

surface = pv.read("model_remeshed.stl")
tet = tetgen.TetGen(surface)
tet.make_manifold()  # built-in repair pass
tet.tetrahedralize(
    order=1,
    mindihedral=10,   # relax
    minratio=2.0,     # relax
    nobisect=True,    # don't insert Steiner points on boundary
)
grid = tet.grid
grid.save("model.vtu")

import meshio
import numpy as np

mesh = meshio.read("model.vtu")

# Add physical/geometrical tags expected by Gmsh/Dolfinx.
# One volume group for tetrahedra and one surface group for boundary triangles.
physical_data = []
geometrical_data = []
field_data = {}

for block in mesh.cells:
    n = len(block.data)
    if block.type == "tetra":
        physical_data.append(np.full(n, 1, dtype=np.int32))
        geometrical_data.append(np.full(n, 1, dtype=np.int32))
        field_data.setdefault("volume", np.array([1, 3], dtype=np.int32))
    elif block.type == "triangle":
        physical_data.append(np.full(n, 2, dtype=np.int32))
        geometrical_data.append(np.full(n, 2, dtype=np.int32))
        field_data.setdefault("boundary", np.array([2, 2], dtype=np.int32))
    else:
        # Keep other element blocks valid with a fallback tag.
        physical_data.append(np.full(n, 99, dtype=np.int32))
        geometrical_data.append(np.full(n, 99, dtype=np.int32))

mesh.cell_data = {
    "gmsh:physical": physical_data,
    "gmsh:geometrical": geometrical_data,
}
mesh.field_data = field_data

meshio.write("model.msh", mesh, file_format="gmsh22", binary=False)