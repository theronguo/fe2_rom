"""Generate a *periodic* 2D hexagonal microstructure (RVE) mesh with gmsh.

Periodicity strategy
--------------------
A regular hexagon has the full dihedral symmetry D6 (12 elements: 6 rotations
of 60 deg + 6 mirrors).  Instead of meshing the whole cell and hoping the mesher
produces matching nodes on opposite edges, we mesh only the **fundamental
domain** -- a 30 deg wedge that is 1/12 of the hexagon -- and then replicate that
single mesh with all 12 symmetry operations.  The assembled mesh therefore has
*exact* D6 symmetry, and that symmetry is precisely what makes opposite edges of
the hexagon carry identical node patterns:

    * the two vertical edges (left/right) are exchanged by the vertical mirror,
      whose action on those edges equals the periodic translation;
    * each slanted edge pair is exchanged by the mirror through the opposite
      pair of vertices, whose action on that edge pair again equals the periodic
      translation.

So a D6-symmetric mesh is automatically periodic on the three edge pairs -- no
affine node matching, no `setPeriodic` needed.  (See `_check_periodicity`, which
verifies it numerically.)

Geometry
--------
Pointy-top regular hexagon, circumradius ``R`` (centre-to-vertex distance ``d``
in the RVE figure).  Vertices at angles 30, 90, ..., 330 deg; the left/right
edges are vertical.  The porous structure is a **hexagonal (triangular-lattice)
packing of equal-radius circular holes** of radius ``r`` -- monodisperse, as in
the reference figure.  The cell contains:

    * one full hole at the centre;
    * one half-hole centred on each of the 6 edge midpoints (these complete into
      full circles when the cell is tiled -- the nearest neighbours of the
      central hole, all at the lattice spacing s = apothem = R*sqrt(3)/2 = ``l``).

That is the 4-hole supercell of the packing (1 centre + 6 * 1/2 edge = 4).  The
material crosses the cell boundary at the 6 vertices, which are the solid
junctions of the packing.  Non-overlap requires r < s/2 = R*sqrt(3)/4.

The fundamental wedge is the triangle O=(0,0), M=(a,0) (edge midpoint),
V=(a, R/2) (vertex), with a = R*sqrt(3)/2 the apothem.  Only the two holes that
touch the wedge -- the central one (at O) and the edge-midpoint one (at M) -- are
subtracted there; the D6 replication rebuilds the rest of the packing.

Usage
-----
    python generate_hexagonal_mesh.py                  # defaults -> hexagonal_rve.msh
    python generate_hexagonal_mesh.py --ell 1.0 --lc 0.02 --preview   # also write .png
    python generate_hexagonal_mesh.py --ell 5.0 --hole-radius 2.25
    python generate_hexagonal_mesh.py --output rve.msh
"""
import argparse
import math
import sys

import numpy as np
import gmsh


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def hexagon_vertices(R):
    """6 vertices of a pointy-top regular hexagon, circumradius R (CCW)."""
    angles = np.deg2rad(30 + 60 * np.arange(6))
    return np.column_stack([R * np.cos(angles), R * np.sin(angles)])


def packing_holes(R, r):
    """Disks (cx, cy, r) to subtract from the fundamental wedge for a hexagonal
    packing of equal circles.

    Two holes touch the 30 deg wedge: the central one at O=(0,0) and the
    edge-midpoint one at M=(a,0), a = R*sqrt(3)/2.  The D6 replication turns these
    into the full central hole and the 6 half-holes on the edge midpoints.
    """
    a = R * math.sqrt(3) / 2.0
    return [(0.0, 0.0, r), (a, 0.0, r)]


# ---------------------------------------------------------------------------
# Step 1: mesh the fundamental domain (1/12 of the hexagon)
# ---------------------------------------------------------------------------
def mesh_fundamental_domain(R, lc, holes):
    """Mesh the 30 deg wedge minus the holes; return (nodes (N,2), tris (M,3))."""
    a = R * math.sqrt(3) / 2.0                        # apothem
    O, M, V = (0.0, 0.0), (a, 0.0), (a, R / 2.0)

    occ = gmsh.model.occ
    gmsh.model.add("fundamental_domain")

    p0 = occ.addPoint(*O, 0.0)
    p1 = occ.addPoint(*M, 0.0)
    p2 = occ.addPoint(*V, 0.0)
    l0 = occ.addLine(p0, p1)                           # O-M : 0 deg ray (x-axis mirror)
    l1 = occ.addLine(p1, p2)                           # M-V : outer hexagon edge (x = a)
    l2 = occ.addLine(p2, p0)                           # V-O : 30 deg ray  (vertex mirror)
    wedge = occ.addPlaneSurface([occ.addCurveLoop([l0, l1, l2])])

    if holes:
        tools = [(2, occ.addDisk(cx, cy, 0.0, r, r)) for (cx, cy, r) in holes]
        occ.cut([(2, wedge)], tools, removeObject=True, removeTool=True)
    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm", 6)         # Frontal-Delaunay
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]
    tag2idx = {int(t): i for i, t in enumerate(tags)}

    tris = None
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    for et, en in zip(etypes, enodes):
        if et == 2:                                    # 3-node triangle
            conn = en.reshape(-1, 3)
            tris = np.vectorize(tag2idx.get)(conn)
    if tris is None:
        raise RuntimeError("no triangles produced in the fundamental domain")
    return coords, tris


# ---------------------------------------------------------------------------
# Step 2: replicate with the 12 symmetry operations of D6
# ---------------------------------------------------------------------------
def d6_operations():
    """The 12 elements of D6 as (2x2 matrix, is_reflection) pairs.

    Generated as rho^k and rho^k . sigma, where rho is a 60 deg rotation and
    sigma is the reflection across the x-axis.  The 12 image wedges tile the
    full 360 deg without gaps or overlaps.
    """
    sigma = np.array([[1.0, 0.0], [0.0, -1.0]])
    ops = []
    for k in range(6):
        th = math.radians(60 * k)
        rot = np.array([[math.cos(th), -math.sin(th)],
                        [math.sin(th),  math.cos(th)]])
        ops.append((rot, False))
        ops.append((rot @ sigma, True))
    return ops


def replicate(nodes, tris):
    """Apply the 12 D6 ops; return stacked (nodes, tris) before node merging."""
    all_nodes, all_tris, offset = [], [], 0
    for mat, reflected in d6_operations():
        all_nodes.append(nodes @ mat.T)
        t = tris + offset
        if reflected:
            t = t[:, [0, 2, 1]]                        # restore CCW orientation
        all_tris.append(t)
        offset += len(nodes)
    return np.vstack(all_nodes), np.vstack(all_tris)


def merge_coincident_nodes(nodes, tris, tol):
    """Weld nodes closer than ``tol`` (the shared wedge boundaries) via union-find."""
    from scipy.spatial import cKDTree

    parent = np.arange(len(nodes))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:                       # path compression
            parent[i], i = root, parent[i]
        return root

    for i, j in cKDTree(nodes).query_pairs(r=tol):
        parent[find(i)] = find(j)

    roots = np.array([find(i) for i in range(len(nodes))])
    uniq, inverse = np.unique(roots, return_inverse=True)
    new_nodes = nodes[uniq]
    new_tris = inverse[tris]

    keep = (new_tris[:, 0] != new_tris[:, 1]) & \
           (new_tris[:, 1] != new_tris[:, 2]) & \
           (new_tris[:, 0] != new_tris[:, 2])
    return new_nodes, new_tris[keep]


# ---------------------------------------------------------------------------
# Periodicity verification
# ---------------------------------------------------------------------------
def _boundary_edges(tris):
    """Edges that belong to exactly one triangle -> the outer/hole boundary."""
    from collections import defaultdict
    count = defaultdict(int)
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            count[(a, b) if a < b else (b, a)] += 1
    return [e for e, c in count.items() if c == 1]


def _count_components(nodes, tris):
    """Number of edge-connected components of the triangle mesh."""
    from collections import defaultdict
    edge2tri = defaultdict(list)
    for ti, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge2tri[(a, b) if a < b else (b, a)].append(ti)

    parent = np.arange(len(tris))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for tlist in edge2tri.values():
        for t in tlist[1:]:
            parent[find(t)] = find(tlist[0])
    return len({find(i) for i in range(len(tris))})


def _check_periodicity(nodes, tris, R, tol):
    """Verify the three hexagon edge pairs carry translation-matched nodes."""
    a = R * math.sqrt(3) / 2.0
    verts = hexagon_vertices(R)                        # V1..V6 at 30,90,...,330

    boundary_idx = np.unique(np.array(_boundary_edges(tris)).ravel())
    bnodes = nodes[boundary_idx]

    def on_segment(pts, q1, q2):
        q1, q2 = np.asarray(q1), np.asarray(q2)
        seg = q2 - q1
        L2 = seg @ seg
        t = np.clip((pts - q1) @ seg / L2, 0.0, 1.0)
        proj = q1 + t[:, None] * seg
        return np.linalg.norm(pts - proj, axis=1) < tol

    # (name, V_a -> V_b edge, V_c -> V_d opposite edge, translation a->b... )
    # edges (CCW): E1 V1-V2, E2 V2-V3, E3 V3-V4, E4 V4-V5, E5 V5-V6, E6 V6-V1
    pairs = [
        ("left<->right ", verts[2], verts[3], verts[5], verts[0]),   # E3 <-> E6
        ("topL<->botR  ", verts[1], verts[2], verts[4], verts[5]),   # E2 <-> E5
        ("topR<->botL  ", verts[0], verts[1], verts[3], verts[4]),   # E1 <-> E4
    ]
    print("  periodicity check (max node mismatch on each edge pair):")
    ok = True
    for name, A1, A2, B1, B2 in pairs:
        A = bnodes[on_segment(bnodes, A1, A2)]
        B = bnodes[on_segment(bnodes, B1, B2)]
        # translation that maps edge A onto edge B (centro-symmetric pairing)
        shift = ((B1 + B2) - (A1 + A2)) / 2.0
        shifted = A + shift
        if len(A) != len(B):
            print(f"    {name}: COUNT MISMATCH  |A|={len(A)} |B|={len(B)}")
            ok = False
            continue
        # nearest-neighbour distance from each shifted-A node to a B node
        from scipy.spatial import cKDTree
        dmax = cKDTree(B).query(shifted)[0].max() if len(B) else float("nan")
        flag = "ok" if dmax < tol else "FAIL"
        print(f"    {name}: {len(A):4d} nodes/edge   max dist = {dmax:.2e}  [{flag}]")
        ok = ok and dmax < tol
    return ok


# ---------------------------------------------------------------------------
# Step 3: write the assembled mesh as a gmsh discrete model
# ---------------------------------------------------------------------------
def write_mesh(nodes, tris, output, msh_version=2.2):
    gmsh.model.add("hexagonal_rve")
    surf = gmsh.model.addDiscreteEntity(2)

    node_tags = np.arange(1, len(nodes) + 1)
    coords3 = np.column_stack([nodes, np.zeros(len(nodes))]).ravel()
    gmsh.model.mesh.addNodes(2, surf, node_tags, coords3)

    tri_tags = np.arange(1, len(tris) + 1)
    gmsh.model.mesh.addElementsByType(surf, 2, tri_tags, (tris + 1).ravel())

    # outer boundary as line elements (handy for BCs / periodic detection later)
    bedges = np.array(_boundary_edges(tris))
    curve = gmsh.model.addDiscreteEntity(1)
    line_tags = np.arange(1, len(bedges) + 1)
    gmsh.model.mesh.addElementsByType(curve, 1, line_tags, (bedges + 1).ravel())

    gmsh.model.addPhysicalGroup(2, [surf], tag=1, name="material")
    gmsh.model.addPhysicalGroup(1, [curve], tag=1, name="boundary")

    gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    gmsh.write(output)


# ---------------------------------------------------------------------------
# Optional matplotlib preview
# ---------------------------------------------------------------------------
def save_preview(nodes, tris, R, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts = hexagon_vertices(R)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(nodes[:, 0], nodes[:, 1], tris, lw=0.25, color="0.3")
    hexloop = np.vstack([verts, verts[0]])
    ax.plot(hexloop[:, 0], hexloop[:, 1], "b--", lw=1.2)
    ax.plot(verts[:, 0], verts[:, 1], "ko", ms=5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"periodic hexagonal RVE  ({len(nodes)} nodes, {len(tris)} tris)")
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ell", type=float, default=1.0,
                   help="apothem l = centre-to-edge-midpoint distance, which is "
                        "the packing spacing (the 'l' arrow in the figure). "
                        "Default 1.0.  Circumradius R = 2*l/sqrt(3).")
    p.add_argument("--lc", type=float, default=0.1,
                   help="target element size (default 0.1)")
    p.add_argument("--hole-radius", type=float, default=0.45,
                   help="radius of the (equal) circular holes (default 0.45). "
                        "Must be < l/2 or the holes overlap.")
    p.add_argument("--output", type=str, default="hexagonal_rve.msh",
                   help="output .msh path (default hexagonal_rve.msh)")
    p.add_argument("--preview", action="store_true",
                   help="also write a PNG preview of the mesh (off by default)")
    return p.parse_args()


def main():
    args = parse_args()
    s = args.ell                                       # packing spacing = apothem l
    R = 2.0 * args.ell / math.sqrt(3.0)                # circumradius
    if args.hole_radius >= s / 2.0:
        raise SystemExit(
            f"hole-radius {args.hole_radius} >= l/2 = {s / 2:.4f}: holes overlap. "
            f"Use a smaller radius (l = {s:.4f}).")
    holes = packing_holes(R, args.hole_radius)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        print(f"hexagonal packing: l={args.ell} (R={R:.4f}), hole r={args.hole_radius}, "
              f"spacing s={s:.4f}, ligament={s - 2 * args.hole_radius:.4f}")
        print(f"1/12 fundamental domain: lc={args.lc}, {len(holes)} holes in wedge")
        nodes, tris = mesh_fundamental_domain(R, args.lc, holes)
        print(f"  wedge mesh: {len(nodes)} nodes, {len(tris)} triangles")

        nodes, tris = replicate(nodes, tris)
        weld_tol = args.lc * 1e-3
        nodes, tris = merge_coincident_nodes(nodes, tris, tol=weld_tol)
        print(f"  full hexagon: {len(nodes)} nodes, {len(tris)} triangles")

        ncomp = _count_components(nodes, tris)
        print(f"  connected components: {ncomp}"
              + ("" if ncomp == 1 else "  <-- WARNING: material is disconnected!"))

        ok = _check_periodicity(nodes, tris, R, tol=10 * weld_tol)
        if not ok:
            print("  WARNING: periodicity check did not pass within tolerance",
                  file=sys.stderr)

        write_mesh(nodes, tris, args.output)
        print(f"Mesh written to {args.output}")

        if args.preview:
            png = args.output.rsplit(".", 1)[0] + ".png"
            save_preview(nodes, tris, R, png)
            print(f"Preview written to {png}")
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
