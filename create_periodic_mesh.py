"""
Generate a periodic mesh of the slab+cross-section geometry.

Periodicity is enforced on the 6 outer faces of the bounding box
(x=±0.6, y=±0.6, z=0/1), which come from Box(100) and Box(101).

Usage:
    python create_periodic_mesh.py            # writes mesh.msh
    python create_periodic_mesh.py out.msh    # custom output file
"""
import sys
import math
import numpy as np
import gmsh

# ---------------------------------------------------------------------------
# Parameters (must match mesh.geo)
# ---------------------------------------------------------------------------
lc          = 0.03
hex_radius  = 0.5
target_area = math.pi * 0.05**2
inner_r     = 0.03
outer_r     = math.sqrt(inner_r**2 + target_area / math.pi)

output_file = sys.argv[1] if len(sys.argv) > 1 else "mesh.msh"

# Bounding-box extents
X0, X1 = -0.6,  0.6   # width  1.2
Y0, Y1 = -0.6,  0.6   # height 1.2
Z0, Z1 =  0.0,  1.0   # depth  1.0

# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("periodic_slab")
occ = gmsh.model.occ

# ---- cross-section volumes --------------------------------------------------
cs_tags = []
for i in range(6):
    angle = i * math.pi / 3
    xc = hex_radius * math.cos(angle)
    yc = hex_radius * math.sin(angle)

    c_out = occ.addCircle(xc, yc, 0, outer_r)
    c_in  = occ.addCircle(xc, yc, 0, inner_r)
    cl_out = occ.addCurveLoop([c_out])
    cl_in  = occ.addCurveLoop([c_in])
    surf   = occ.addPlaneSurface([cl_out, cl_in])
    extruded = occ.extrude([(2, surf)], 0, 0, 1)
    # extrude returns [(2,top), (3,vol), (2,side1), ...]
    vol_tag = next(t for dim, t in extruded if dim == 3)
    cs_tags.append(vol_tag)

# ---- two slabs -------------------------------------------------------------
slab_1 = occ.addBox(X0, Y0, 0.0, X1-X0, Y1-Y0, 0.1)
slab_2 = occ.addBox(X0, Y0, 0.9, X1-X0, Y1-Y0, 0.1)

# ---- boolean union ---------------------------------------------------------
tool_tags  = [(3, t) for t in cs_tags]
obj_tags   = [(3, slab_1), (3, slab_2)]
merged, _  = occ.fuse(obj_tags, tool_tags, removeObject=True, removeTool=True)
occ.synchronize()

# Merged volumes → physical group
vol_tags = [t for dim, t in merged if dim == 3]
gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="domain")

# ---------------------------------------------------------------------------
# Identify the 6 outer face groups by bounding box
# ---------------------------------------------------------------------------
tol = lc  # generous; outer faces span the full width/height

all_surfaces = gmsh.model.getEntities(2)

faces = {d: [] for d in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

for _, stag in all_surfaces:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, stag)
    if abs(xmin - X0) < tol and abs(xmax - X0) < tol:
        faces["xmin"].append(stag)
    elif abs(xmin - X1) < tol and abs(xmax - X1) < tol:
        faces["xmax"].append(stag)
    elif abs(ymin - Y0) < tol and abs(ymax - Y0) < tol:
        faces["ymin"].append(stag)
    elif abs(ymin - Y1) < tol and abs(ymax - Y1) < tol:
        faces["ymax"].append(stag)
    elif abs(zmin - Z0) < tol and abs(zmax - Z0) < tol:
        faces["zmin"].append(stag)
    elif abs(zmin - Z1) < tol and abs(zmax - Z1) < tol:
        faces["zmax"].append(stag)

for key, tags in faces.items():
    print(f"  {key}: surface tags {tags}")

# ---------------------------------------------------------------------------
# Periodic surface constraints
#   slave = master translated by (tx, ty, tz)
#   affine transform (row-major 4×4 flattened):
#     [1 0 0 tx | 0 1 0 ty | 0 0 1 tz | 0 0 0 1]
# ---------------------------------------------------------------------------
def translation_matrix(tx, ty, tz):
    return [1, 0, 0, tx,
            0, 1, 0, ty,
            0, 0, 1, tz,
            0, 0, 0, 1]

dx = X1 - X0   # 1.2
dy = Y1 - Y0   # 1.2
dz = Z1 - Z0   # 1.0

periodic_pairs = [
    # (slave_face, master_face, translation_from_master_to_slave)
    ("xmin", "xmax", (-dx,   0,   0)),
    ("ymin", "ymax", (  0, -dy,   0)),
    ("zmin", "zmax", (  0,   0, -dz)),
]

for slave_key, master_key, (tx, ty, tz) in periodic_pairs:
    slave_tags  = faces[slave_key]
    master_tags = faces[master_key]
    if not slave_tags or not master_tags:
        print(f"WARNING: missing faces for pair ({slave_key}, {master_key}) — skipping")
        continue
    gmsh.model.mesh.setPeriodic(2, slave_tags, master_tags,
                                translation_matrix(tx, ty, tz))
    print(f"  Periodic: {slave_key}{slave_tags} <- {master_key}{master_tags}  "
          f"translate=({tx},{ty},{tz})")

# ---------------------------------------------------------------------------
# Mesh options and generation
# ---------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

gmsh.model.mesh.generate(3)

gmsh.write(output_file)
print(f"Mesh written to {output_file}")

gmsh.finalize()
