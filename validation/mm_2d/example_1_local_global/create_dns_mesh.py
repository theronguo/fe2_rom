"""Generate a fully periodic DNS mesh for an nx·ℓ × ny·ℓ plate.

Construction:
  1. Build a single ℓ × ℓ unit cell with one circular hole (diameter d = 8.67 mm).
  2. Copy it nx × ny times to tile the full plate.
  3. Fragment all tiles to merge shared internal edges (conformal interfaces).
  4. Apply periodic mesh constraints: left ↔ right, bottom ↔ top.

By default the domain is [0, W] × [0, H].  Pass --center to shift it to
[−W/2, W/2] × [−H/2, H/2], which is required when the mesh is used as an
RVE in the micromorphic solver (the ansatz assumes X = 0 at the cell centre).
With --nx 2 --ny 2 --center the domain is [−ℓ, ℓ]², matching create_rve_mesh.py.

Physical groups:
    domain (surface, tag 1)
    bottom (curve, tag 1) — y = Y0
    top    (curve, tag 2) — y = Y1
    left   (curve, tag 3) — x = X0
    right  (curve, tag 4) — x = X1

Usage:
    python create_dns_mesh.py
    python create_dns_mesh.py --nx 6 --ny 30
    python create_dns_mesh.py --nx 2 --ny 2 --center --output rve.msh
    python create_dns_mesh.py --output myfile.msh
"""
import argparse
import os

import gmsh

ELL = 9.97      # mm   (unit-cell edge length, paper "ℓ")
HOLE_D = 8.67   # mm   (hole diameter, paper "d")
HOLE_R = 0.5 * HOLE_D


def translation_matrix(tx, ty, tz=0.0):
    """Row-major 4×4 affine matrix for a pure translation."""
    return [1, 0, 0, tx,
            0, 1, 0, ty,
            0, 0, 1, tz,
            0, 0, 0, 1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nx", type=int, default=6,
                   help="number of unit cells in x (default 6)")
    p.add_argument("--ny", type=int, default=30,
                   help="number of unit cells in y (default 30)")
    p.add_argument("--center", action="store_true",
                   help="centre the domain at the origin: [−W/2,W/2]×[−H/2,H/2]")
    p.add_argument("--h", type=float, default=10.0,
                   help="element size divisor: lc = ℓ/h (default 10, i.e. lc = ℓ/10)")
    p.add_argument("--output", type=str, default=None,
                   help="output .msh path (default ./dns_periodic_<nx>x<ny>.msh)")
    return p.parse_args()


def main():
    args = parse_args()
    nx, ny = args.nx, args.ny
    W = nx * ELL
    H = ny * ELL
    lc = ELL / args.h
    X0 = -W / 2 if args.center else 0.0
    Y0 = -H / 2 if args.center else 0.0
    X1 = X0 + W
    Y1 = Y0 + H
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"dns_{nx}x{ny}.msh",
    )

    gmsh.initialize()
    gmsh.model.add(f"dns_{nx}x{ny}")
    occ = gmsh.model.occ

    # --- Build the unit cell at the grid origin, then tile -------------------
    # The unit cell always starts at (X0, Y0) so the full tiled domain spans
    # [X0, X1] × [Y0, Y1] regardless of the --center flag.
    cell_rect = occ.addRectangle(X0, Y0, 0.0, ELL, ELL)
    cell_hole = occ.addDisk(X0 + ELL / 2, Y0 + ELL / 2, 0.0, HOLE_R, HOLE_R)
    cell_out, _ = occ.cut(
        [(2, cell_rect)], [(2, cell_hole)],
        removeObject=True, removeTool=True,
    )
    occ.synchronize()
    unit_tag = cell_out[0][1]

    # --- Tile: copy unit cell to fill the nx × ny grid -----------------------
    all_cells = [(2, unit_tag)]
    for k in range(nx):
        for j in range(ny):
            if k == 0 and j == 0:
                continue
            copies = occ.copy([(2, unit_tag)])
            occ.translate(copies, k * ELL, j * ELL, 0.0)
            all_cells.extend(copies)

    # Fragment merges all shared internal edges → conformal interface mesh.
    out_ents, _ = occ.fragment(all_cells, [], removeObject=True, removeTool=True)
    occ.synchronize()

    # --- Physical groups ------------------------------------------------------
    surf_tags = [t for d, t in out_ents if d == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="domain")

    tol = 1e-4 * ELL
    curves = {"xmin": [], "xmax": [], "ymin": [], "ymax": []}
    for _, ctag in gmsh.model.getEntities(1):
        xlo, ylo, _, xhi, yhi, _ = gmsh.model.getBoundingBox(1, ctag)
        if abs(xlo - X0) < tol and abs(xhi - X0) < tol:
            curves["xmin"].append(ctag)
        elif abs(xlo - X1) < tol and abs(xhi - X1) < tol:
            curves["xmax"].append(ctag)
        elif abs(ylo - Y0) < tol and abs(yhi - Y0) < tol:
            curves["ymin"].append(ctag)
        elif abs(ylo - Y1) < tol and abs(yhi - Y1) < tol:
            curves["ymax"].append(ctag)

    for key, tags in curves.items():
        if not tags:
            raise RuntimeError(f"No curves found for boundary '{key}'.")
        print(f"  {key}: {len(tags)} curve(s) — tags {tags[:5]}{'...' if len(tags) > 5 else ''}")

    gmsh.model.addPhysicalGroup(1, curves["ymin"], tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, curves["ymax"], tag=2, name="top")
    gmsh.model.addPhysicalGroup(1, curves["xmin"], tag=3, name="left")
    gmsh.model.addPhysicalGroup(1, curves["xmax"], tag=4, name="right")

    # --- Periodic mesh constraints -------------------------------------------
    # slave = master translated by affine; master nodes are reused on slave.
    # xmin (x=0) ← xmax (x=W) by translation (-W, 0)
    # ymin (y=0) ← ymax (y=H) by translation (0, -H)
    gmsh.model.mesh.setPeriodic(
        1, curves["xmin"], curves["xmax"], translation_matrix(-W, 0.0)
    )
    gmsh.model.mesh.setPeriodic(
        1, curves["ymin"], curves["ymax"], translation_matrix(0.0, -H)
    )

    # --- Mesh options ---------------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(out)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"  x ∈ [{X0:.3f}, {X1:.3f}] mm   y ∈ [{Y0:.3f}, {Y1:.3f}] mm   holes = {nx * ny}   lc = {lc:.4f} mm (ℓ/{args.h:g})")
    print(f"  nodes ≈ {n_nodes}")
    print(f"Mesh written to {out}")

    gmsh.finalize()


if __name__ == "__main__":
    main()
