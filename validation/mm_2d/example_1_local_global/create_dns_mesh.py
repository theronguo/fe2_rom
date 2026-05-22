"""Generate the DNS mesh for van Bree et al. (2020), Section 4.1.

Geometry:
  - Rectangular specimen of width W = w·ℓ and height H = h·ℓ (defaults
    w = 6, h = 30 to reproduce the reference 6ℓ × 30ℓ realization of Fig. 5a).
  - Square stacking of circular holes (diameter d = 8.67 mm, spacing ℓ).
    Holes are centred at ((k + 0.5)·ℓ, (j + 0.5)·ℓ) for k = 0..w−1, j = 0..h−1
    — origin is the bottom-left corner.

Mesh:
  - Isoparametric quadratic (P2, 6-node) triangles.
  - Element size h_m = ℓ/10 (same as the RVE mesh, paper Fig. 4b).
  - Physical groups:
        domain (surface, tag 1)
        bottom (curve, tag 1) — y = 0
        top    (curve, tag 2) — y = H
  - Saved as Gmsh format 4.x (default), which the dolfinx `io.gmsh`
    reader consumes; physical groups are required for facet tagging.

Usage:
    python create_dns_mesh.py                       # 6ℓ × 30ℓ, dns.msh
    python create_dns_mesh.py --w 4 --h 8           # match macro spec (4ℓ × 8ℓ)
    python create_dns_mesh.py --output mydns.msh
"""
import argparse
import os

import gmsh

# ---------------------------------------------------------------------------
ELL = 9.97          # mm
HOLE_D = 8.67       # mm
HOLE_R = 0.5 * HOLE_D
LC = ELL / 10.0     # mm   (h_m = ℓ/10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--w", type=int, default=6,
                   help="specimen width in units of ℓ (default 6)")
    p.add_argument("--h", type=int, default=30,
                   help="specimen height in units of ℓ (default 30)")
    p.add_argument("--output", type=str, default=None,
                   help="output .msh path (default ./dns_<w>x<h>.msh)")
    return p.parse_args()


def main():
    args = parse_args()
    W = args.w * ELL
    H = args.h * ELL
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"dns_{args.w}x{args.h}.msh"
    )

    gmsh.initialize()
    gmsh.model.add(f"dns_{args.w}x{args.h}")
    occ = gmsh.model.occ

    rect_tag = occ.addRectangle(0.0, 0.0, 0.0, W, H)

    hole_tags = []
    for k in range(args.w):
        for j in range(args.h):
            cx = (k + 0.5) * ELL
            cy = (j + 0.5) * ELL
            hole_tags.append(occ.addDisk(cx, cy, 0.0, HOLE_R, HOLE_R))

    # Cut all holes at once (boolean fragmentation is faster than sequential cuts).
    out_ents, _ = occ.cut(
        [(2, rect_tag)],
        [(2, t) for t in hole_tags],
        removeObject=True, removeTool=True,
    )
    occ.synchronize()

    # Physical groups -------------------------------------------------------
    surf_tags = [t for d, t in out_ents if d == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=1, name="domain")

    # Tag outer-boundary curves at y = 0 and y = H by bounding-box.
    tol = 1e-4 * ELL
    bot_curves, top_curves = [], []
    for _, ctag in gmsh.model.getEntities(1):
        _, ymin, _, _, ymax, _ = gmsh.model.getBoundingBox(1, ctag)
        if abs(ymin) < tol and abs(ymax) < tol:
            bot_curves.append(ctag)
        elif abs(ymin - H) < tol and abs(ymax - H) < tol:
            top_curves.append(ctag)

    if not bot_curves or not top_curves:
        raise RuntimeError(
            f"Failed to identify top/bottom curves "
            f"(found {len(bot_curves)} bottom, {len(top_curves)} top)."
        )
    gmsh.model.addPhysicalGroup(1, bot_curves, tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, top_curves, tag=2, name="top")

    # Mesh options ----------------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", LC)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", LC)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.mesh.generate(2)
    gmsh.write(out)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"  W = {W:.3f} mm   H = {H:.3f} mm   holes = {args.w * args.h}")
    print(f"  nodes ≈ {n_nodes}")
    print(f"Mesh written to {out}")

    gmsh.finalize()


if __name__ == "__main__":
    main()
