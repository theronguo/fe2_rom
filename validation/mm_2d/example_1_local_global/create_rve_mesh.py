"""Generate the 2ℓ × 2ℓ periodic RVE mesh of Fig. 4(b) (van Bree et al. 2020).

Geometry:
  - Domain: square [-l, l] × [-l, l] with l = 9.97 mm (so width = 2l).
  - Four circular holes of diameter d = 8.67 mm centered at (±l/2, ±l/2)
    — square stacking with hole-to-hole spacing l.

Mesh:
  - Quadratic (P2, 6-node) triangles.
  - Characteristic element size h_m = l/10.
  - Periodic in both directions: the 1D boundary curves on x = -l are tied to
    those on x = +l (translation (+2l, 0)), likewise for y.
  - Saved as Gmsh format 2.2, which is what the codebase's ``read_from_msh``
    path consumes.

Usage:
    python create_rve_mesh.py              # writes rve.msh next to this file
    python create_rve_mesh.py custom.msh   # custom output path
"""
import os
import sys

import gmsh

# ---------------------------------------------------------------------------
# Parameters (paper Section 4.1)
# ---------------------------------------------------------------------------
ELL = 9.97          # mm   (half-cell edge length, paper "ℓ")
HOLE_D = 8.67       # mm   (hole diameter, paper "d")
HOLE_R = 0.5 * HOLE_D
LC = ELL / 10.0     # mm   (characteristic element size, paper "h_m = ℓ/10")

X0, X1 = -ELL,  ELL
Y0, Y1 = -ELL,  ELL

HERE = os.path.dirname(os.path.abspath(__file__))
output_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rve.msh")

# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("rve_square_4holes")
occ = gmsh.model.occ

# Outer rectangle (as a surface).
rect_tag = occ.addRectangle(X0, Y0, 0.0, X1 - X0, Y1 - Y0)

# Four holes at (±l/2, ±l/2).
hole_centers = [(+ELL / 2, +ELL / 2),
                (-ELL / 2, +ELL / 2),
                (+ELL / 2, -ELL / 2),
                (-ELL / 2, -ELL / 2)]
hole_tags = [occ.addDisk(cx, cy, 0.0, HOLE_R, HOLE_R) for cx, cy in hole_centers]

# Cut: rectangle - holes.
out, _ = occ.cut([(2, rect_tag)], [(2, t) for t in hole_tags],
                 removeObject=True, removeTool=True)
occ.synchronize()

surf_tags = [t for d, t in out if d == 2]
gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="domain")

# ---------------------------------------------------------------------------
# Identify outer boundary curves by bounding-box tagging.
# (Hole boundary curves stay interior to the periodicity statement; only the
# four outer edges of the square are tied.)
# ---------------------------------------------------------------------------
tol = 1e-4 * ELL
all_curves = gmsh.model.getEntities(1)
curves = {"xmin": [], "xmax": [], "ymin": [], "ymax": []}

for _, ctag in all_curves:
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
    if abs(xmin - X0) < tol and abs(xmax - X0) < tol:
        curves["xmin"].append(ctag)
    elif abs(xmin - X1) < tol and abs(xmax - X1) < tol:
        curves["xmax"].append(ctag)
    elif abs(ymin - Y0) < tol and abs(ymax - Y0) < tol:
        curves["ymin"].append(ctag)
    elif abs(ymin - Y1) < tol and abs(ymax - Y1) < tol:
        curves["ymax"].append(ctag)

for key, tags in curves.items():
    print(f"  {key}: curve tags {tags}")

# ---------------------------------------------------------------------------
# Periodic curve constraints: slave = master translated by (tx, ty, 0).
#   affine = [1 0 0 tx, 0 1 0 ty, 0 0 1 0, 0 0 0 1]   (row-major flattened)
# ---------------------------------------------------------------------------
def translation_matrix(tx, ty, tz=0.0):
    return [1, 0, 0, tx,
            0, 1, 0, ty,
            0, 0, 1, tz,
            0, 0, 0, 1]

dx = X1 - X0   # 2ℓ
dy = Y1 - Y0   # 2ℓ

periodic_pairs = [
    # (slave, master, translation from master to slave)
    ("xmin", "xmax", (-dx,   0)),
    ("ymin", "ymax", (  0, -dy)),
]
for s_key, m_key, (tx, ty) in periodic_pairs:
    s_tags = curves[s_key]
    m_tags = curves[m_key]
    if not s_tags or not m_tags:
        raise RuntimeError(f"Missing periodic-pair curves for ({s_key}, {m_key}).")
    gmsh.model.mesh.setPeriodic(1, s_tags, m_tags, translation_matrix(tx, ty))
    print(f"  Periodic: {s_key}{s_tags} <- {m_key}{m_tags}  translate=({tx},{ty})")

# ---------------------------------------------------------------------------
# Mesh options.
# ---------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)
gmsh.option.setNumber("Mesh.ElementOrder", 2)            # quadratic triangles
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)   # full P2 (6-node)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)        # required by read_from_msh

gmsh.model.mesh.generate(2)

gmsh.write(output_file)
print(f"Mesh written to {output_file}")

gmsh.finalize()
