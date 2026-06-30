"""Generate a periodic, symmetric 3D *simple-cubic* corner-node lattice mesh.

Topology (spheres on the cell corners; "BCC without the centre node")
---------------------------------------------------------------------
A spherical node on every cube corner, joined to its three corner neighbours
by straight circular (cylindrical) struts along the cube edges:

    * 8 spherical nodes of radius `rn` at the corners (+-half, +-half, +-half);
    * 12 circular struts of radius `rs` along the cube edges connecting adjacent
      corners.

Per unit cell this is 1 node + 3 struts worth of material (each corner is shared
by 8 cells, each edge by 4).  There is *no* central node and *no* body diagonal
-- it is the body-centred-cubic lattice with the centre removed.  Loaded in
compression along an axis the edge struts carry the load and -- if slender
(small `rs`) -- buckle, with the spheres acting as joints.

Node placement & periodicity (why the corner is shared 1/8 per cell)
--------------------------------------------------------------------
``reflect_full`` mirrors the octant across the cell *mid-planes* x=0, y=0, z=0.
A sphere centred on the corner (half, half, half) is therefore cut to 1/8 inside
the octant; the three edge struts leaving that corner clip to quarter-cylinders.
Reflecting 8x places an eighth-sphere on each of the 8 corners and a
quarter-strut on each of the 12 edges -- the correct periodic unit cell.  When
*tiled*, the 8 eighth-spheres meeting at each interior corner weld into a full
sphere, and the 4 quarter-struts meeting at each interior edge weld into a full
strut, so a multi-cell DNS shows complete spheres joined by complete struts.

Construction by reflection (as in create_mesh.py)
-------------------------------------------------
    1. build the full cell geometry (8 corner spheres + 12 edge struts), fuse;
    2. intersect with the positive octant [0, half]^3 -> octant solid (1/8 node
       sphere + three quarter-struts) whose inner faces lie on x=0, y=0, z=0;
    3. mesh the octant once;
    4. reflect that octant mesh across the three mid-planes (8 sign copies),
       welding the coincident plane nodes.

Because the cell is *assembled by reflection*, it is symmetric node-for-node
about the three mid-planes, and opposite cell faces carry identical node
patterns -> the cell tiles conformally (periodic).

Containment / centring
----------------------
The single unit cell fits in [-half, half]^3 (eighth-spheres and quarter-struts
straddle the corners/edges of the box).  The tiled domain defaults to the DNS
box [0, Nx] x [0, Ny] x [0, Nz]; ``--center`` shifts it to the origin-centred
RVE box [-Nx/2, Nx/2] x ... (for Nx=Ny=Nz=1 this is the [-0.5, 0.5]^3 RVE).

Usage
-----
    python create_sphere_strut_mesh.py                       # cell on [0,1]^3 -> sphere_strut.msh
    python create_sphere_strut_mesh.py --center              # RVE on [-0.5,0.5]^3
    python create_sphere_strut_mesh.py --nx 3 --ny 3 --nz 3  # 3x3x3 DNS (full spheres+struts)
    python create_sphere_strut_mesh.py --nx 2 --ny 2 --nz 2 --center
    python create_sphere_strut_mesh.py --strut-radius 0.05 --lc 0.02
    python create_sphere_strut_mesh.py --node-radius 0.16    # set the node radius directly
    python create_sphere_strut_mesh.py --octant-only         # write just the eighth
    python create_sphere_strut_mesh.py --nx 2 --ny 2 --nz 2 --solid-skin 0.15
                                                             # DNS lattice inside a solid box skin

Solid outer skin (DNS only)
---------------------------
``--solid-skin T`` wraps the whole DNS box [0,Nx]x[0,Ny]x[0,Nz] in a closed
solid shell of thickness ``T`` on all six faces, with the lattice filling the
interior.  This is *not* compatible with ``--center`` (it is a finite skinned
specimen, not a periodic RVE) and bypasses the reflect/tile pipeline for a full
gmsh-OCC boolean build (lattice cells fused with the box shell, then clipped to
the box), so it is conformal but considerably slower -- keep Nx*Ny*Nz modest.
"""
import argparse
import os

import numpy as np
import gmsh

def tile(coords, tets, nx, ny, nz, cell):
    """Repeat a unit cell (centred at origin, edge `cell`) on an Nx*Ny*Nz grid.

    Returns the tiled mesh on [0, Nx*cell] x [0, Ny*cell] x [0, Nz*cell].
    Pure translation preserves orientation, so no repair is needed.
    """
    n = len(coords)
    blocks_c, blocks_t, off = [], [], 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = (np.array([i, j, k]) + 0.5) * cell
                blocks_c.append(coords + shift)
                blocks_t.append(tets + off)
                off += n
    return weld(np.vstack(blocks_c), np.vstack(blocks_t))

def weld(coords, tets, merge_tol=1e-7):
    """Merge coincident nodes (by rounded coordinate) and remap connectivity.

    Shared nodes carry *identical* float coordinates by construction (exact
    sign flips / integer translations), so the rounding only collapses true
    duplicates, never distinct-but-close nodes.
    """
    keys = np.round(coords / merge_tol).astype(np.int64)
    _, idx, inv = np.unique(keys, axis=0, return_index=True,
                            return_inverse=True)
    return coords[idx], inv.reshape(-1)[tets]

VOL_TAG = 1     # "volume"   (dim 3)
BND_TAG = 2     # "boundary" (dim 2)

# --- element layout for linear (4-node) and quadratic (10-node) tets --------
# gmsh second-order tet (MSH type 11) node order: 4 vertices, then the six edge
# midpoints of edges (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).  Swapping the first
# two vertices (to repair a negative signed volume) permutes both the vertices
# and the edge nodes that ride along with them:
TET_FLIP = {4: [1, 0, 2, 3],
            10: [1, 0, 2, 3, 4, 6, 5, 8, 7, 9]}
# Faces of a tet for boundary extraction.  The first three entries of each row
# are the corner triple (used to match faces); quadratic faces are 6-node tris
# [c0,c1,c2, m01,m12,m20].
TET_FACES = {
    4:  [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    10: [[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
         [0, 2, 3, 6, 9, 7], [1, 2, 3, 5, 9, 8]],
}
TET_TYPE = {4: 4, 10: 11}       # MSH code: linear / quadratic tetrahedron
TRI_TYPE = {3: 2, 6: 9}         # MSH code: linear / quadratic triangle


def fix_orientation(coords, tets):
    """Swap nodes of any tet with negative signed volume (linear or quadratic).

    Orientation is decided from the four corner nodes (slots 0-3); for
    quadratic tets the edge nodes are permuted along with the swapped corners.
    """
    p = coords[tets[:, :4]]                          # (Nt, 4, 3) corners only
    v = np.einsum("ij,ij->i",
                  p[:, 1] - p[:, 0],
                  np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
    flip = v < 0.0
    tets[flip] = tets[flip][:, TET_FLIP[tets.shape[1]]]
    return tets


def reflect_full(coords, tets):
    """Assemble one unit cell by reflecting the octant across x,y,z = 0.

    Nodes on a symmetry plane are shared between the straddling copies and get
    merged; element orientation is repaired afterwards.  Works for linear and
    curved tets — curved mid-edge nodes reflect with the (symmetric) geometry.
    """
    n = len(coords)
    blocks_c, blocks_t, off = [], [], 0
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                blocks_c.append(coords * np.array([sx, sy, sz]))
                blocks_t.append(tets + off)
                off += n
    coords_full, tets_full = weld(np.vstack(blocks_c), np.vstack(blocks_t))
    return coords_full, fix_orientation(coords_full, tets_full)


def boundary_triangles(tets):
    """Return the (3- or 6-node) tet faces that belong to exactly one tet."""
    face_defs = np.array(TET_FACES[tets.shape[1]])
    fw = face_defs.shape[1]                           # 3 (linear) or 6 (quad)
    faces = tets[:, face_defs].reshape(-1, fw)
    keys = np.sort(faces[:, :3], axis=1)              # corner triple = face id
    _, idx, counts = np.unique(keys, axis=0, return_index=True,
                               return_counts=True)
    return faces[idx[counts == 1]]


def write_msh(path, coords, tets, tris):
    """Write a gmsh v2.2 ASCII mesh with 'volume' (3D) and 'boundary' (2D).

    Element types follow the connectivity width, so both linear (4-node tet /
    3-node tri) and quadratic (10-node tet / 6-node tri) meshes are supported.
    """
    n_nodes = len(coords)
    n_tri, n_tet = len(tris), len(tets)
    tet_type, tri_type = TET_TYPE[tets.shape[1]], TRI_TYPE[tris.shape[1]]

    node_block = np.column_stack([np.arange(1, n_nodes + 1), coords])
    tri_block = np.column_stack([
        np.arange(1, n_tri + 1),
        np.full(n_tri, tri_type), np.full(n_tri, 2),
        np.full(n_tri, BND_TAG), np.full(n_tri, BND_TAG),
        tris + 1,
    ])
    tet_block = np.column_stack([
        np.arange(n_tri + 1, n_tri + n_tet + 1),
        np.full(n_tet, tet_type), np.full(n_tet, 2),
        np.full(n_tet, VOL_TAG), np.full(n_tet, VOL_TAG),
        tets + 1,
    ])

    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n2\n")
        f.write(f'2 {BND_TAG} "boundary"\n')
        f.write(f'3 {VOL_TAG} "volume"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{n_nodes}\n")
        np.savetxt(f, node_block, fmt="%d %.16e %.16e %.16e")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{n_tri + n_tet}\n")
        np.savetxt(f, tri_block, fmt="%d")
        np.savetxt(f, tet_block, fmt="%d")
        f.write("$EndElements\n")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_corner_lattice_volume(occ, half, rs, rn):
    """Build the cell (8 corner spheres + 12 edge struts) and fuse it.

    Returns the fused volume dim-tags (a single solid).
    """
    c = half
    signs = (-1.0, 1.0)
    corners = [(sx * c, sy * c, sz * c)
               for sx in signs for sy in signs for sz in signs]

    vols = [occ.addSphere(cx, cy, cz, rn) for cx, cy, cz in corners]

    # 12 cube-edge struts: corner pairs differing in exactly one axis
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            ci, cj = np.array(corners[i]), np.array(corners[j])
            d = cj - ci
            if (np.abs(d) > 1e-12).sum() == 1:        # adjacent along one edge
                vols.append(
                    occ.addCylinder(ci[0], ci[1], ci[2], d[0], d[1], d[2], rs))

    obj = [(3, vols[0])]
    tool = [(3, t) for t in vols[1:]]
    fused, _ = occ.fuse(obj, tool, removeObject=True, removeTool=True)
    occ.synchronize()
    return [t for d, t in fused if d == 3]


def mesh_octant(half, rs, rn, lc, order=1):
    """Build + mesh one octant of the corner lattice. Returns (coords, tets).

    ``order=2`` produces curved (10-node) tets whose mid-edge nodes are
    projected onto the curved sphere/strut CAD surfaces, so the round geometry
    is captured with far fewer elements than straight-faced linear tets.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("corner_lattice_octant")
    occ = gmsh.model.occ

    vols = build_corner_lattice_volume(occ, half, rs, rn)

    # Octant cut box [0, half]^3.  Inner faces (x=y=z=0) become the symmetry
    # planes for reflection; the octant holds the 1/8 corner sphere plus the
    # three quarter-struts running inward along the cube edges.
    octbox = occ.addBox(0.0, 0.0, 0.0, half, half, half)
    occ.intersect([(3, t) for t in vols], [(3, octbox)],
                  removeObject=True, removeTool=True)
    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
    if order >= 2:
        # 0 = curve the high-order nodes onto the geometry (default); be
        # explicit, and optimise so curved tets near tight struts don't invert.
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)

    node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_xyz.reshape(-1, 3)
    # gmsh node tags are 1-based but not necessarily contiguous -> remap
    tag2row = {int(t): i for i, t in enumerate(node_tags)}

    etype = 11 if order >= 2 else 4         # 11 = 10-node quad tet, 4 = linear
    nnodes = 10 if order >= 2 else 4
    _, enodes = gmsh.model.mesh.getElementsByType(etype)
    enodes = enodes.reshape(-1, nnodes)
    tets = np.vectorize(tag2row.get)(enodes).astype(np.int64)

    gmsh.finalize()
    return node_xyz, tets


def mesh_full_cell_order2(half, rs, rn, lc):
    """Mesh one full periodic unit cell directly at order 2 (no reflection).

    The reflect/tile path mirrors a meshed octant across the cell mid-planes;
    that works for straight (linear) tets but *inverts* the curved (10-node)
    tets of an order-2 mesh — reflection flips orientation, and while the
    corner-node swap restores a positive *linear* volume the curved cell map
    still folds (≈10% of cells get a negative Jacobian, so the mesh will not
    solve).  Instead we mesh the whole cell in a single gmsh pass and tie the
    three opposite face pairs with ``setPeriodic``, so the curved tets are
    valid (positive Jacobian) *and* opposite faces are node-coincident — both
    required for periodic homogenization.

    Returns ``(coords, tets)`` for one cell centred at the origin in
    ``[-half, half]^3``, matching ``reflect_full`` so the downstream
    ``tile`` / boundary / writer pipeline is unchanged (pure translation keeps
    the curved cells valid when tiling a multi-cell DNS).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("corner_lattice_cell")
    occ = gmsh.model.occ

    cell = 2.0 * half
    vols = build_corner_lattice_volume(occ, half, rs, rn)
    box = occ.addBox(-half, -half, -half, cell, cell, cell)
    occ.intersect([(3, t) for t in vols], [(3, box)],
                  removeObject=True, removeTool=True)
    occ.synchronize()

    # Tie each pair of opposite box faces (slave = +face, master = -face,
    # translated by one cell) so the boundary mesh is periodic node-for-node.
    eps = 1e-6 * cell
    def faces_at(axis, value):
        lo = [-half - eps, -half - eps, -half - eps]
        hi = [half + eps, half + eps, half + eps]
        lo[axis], hi[axis] = value - eps, value + eps
        return [t for _, t in gmsh.model.getEntitiesInBoundingBox(*lo, *hi, 2)]
    for axis in range(3):
        minus, plus = faces_at(axis, -half), faces_at(axis, half)
        if minus and plus:
            affine = np.eye(4)
            affine[axis, 3] = cell
            gmsh.model.mesh.setPeriodic(2, plus, minus, affine.flatten().tolist())

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
    gmsh.model.mesh.generate(3)

    node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_xyz.reshape(-1, 3)
    tag2row = {int(t): i for i, t in enumerate(node_tags)}
    _, enodes = gmsh.model.mesh.getElementsByType(11)   # 11 = 10-node quad tet
    enodes = enodes.reshape(-1, 10)
    tets = np.vectorize(tag2row.get)(enodes).astype(np.int64)

    gmsh.finalize()
    return node_xyz, tets


def mesh_solid_skin_dns(nx, ny, nz, half, rs, rn, skin, lc):
    """Build a DNS lattice enclosed in a solid outer skin of thickness `skin`.

    The lattice is tiled over the box [0, W] x [0, H] x [0, D] (W = nx*cell,
    etc.).  The tiled lattice is clipped to the inset *inner* box (inset by
    `skin` on every face) and fragmented against the solid shell that fills
    box \\ inner_box, so the six outer faces are solid (thickness `skin`) and the
    lattice fills the interior; struts reaching the inner-box planes bond into
    the skin.  Returns (coords, tets) just like mesh_octant.

    Clipping to the *inner* box keeps the only lattice/skin interface on flat
    planes, and ``fragment`` (rather than ``fuse``) makes that interface a
    conformal shared surface -- both are needed for tetgen to mesh the curved
    lattice reliably.  All resulting sub-volumes are one material; the numpy
    writer keys off element type, and internal faces drop out of the boundary
    because they are shared by two tets.

    Unlike the reflect/tile pipeline this is a full OCC boolean build, so it is
    considerably slower and meant for modest Nx*Ny*Nz; periodicity is neither
    available nor needed for a finite skinned specimen.
    """
    cell = 2.0 * half
    W, H, D = nx * cell, ny * cell, nz * cell

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("sphere_strut_solid_skin")
    occ = gmsh.model.occ

    # 1. one origin-centred unit cell (template), copied onto the Nx*Ny*Nz grid,
    #    then fused into a single tiled lattice solid.
    template = [(3, t) for t in build_corner_lattice_volume(occ, half, rs, rn)]
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cp = occ.copy(template)
                occ.translate(cp, (i + 0.5) * cell,
                              (j + 0.5) * cell, (k + 0.5) * cell)
                cells.extend(cp)
    occ.synchronize()
    occ.remove(template, recursive=True)          # drop the untranslated template
    if len(cells) > 1:
        lattice, _ = occ.fuse([cells[0]], cells[1:],
                              removeObject=True, removeTool=True)
    else:
        lattice = cells
    occ.synchronize()

    # 2. clip the lattice to the inset inner box (interior only); keep the box.
    inner = occ.addBox(skin, skin, skin,
                       W - 2 * skin, H - 2 * skin, D - 2 * skin)
    lat_in, _ = occ.intersect(lattice, [(3, inner)],
                              removeObject=True, removeTool=False)
    occ.synchronize()

    # 3. solid skin shell = outer box minus that same inner box.
    outer = occ.addBox(0.0, 0.0, 0.0, W, H, D)
    shell, _ = occ.cut([(3, outer)], [(3, inner)],
                       removeObject=True, removeTool=True)
    occ.synchronize()

    # 4. fragment lattice + shell -> conformal shared interface on the inner-box
    #    planes; weld coincident boundaries.  All sub-volumes are one material.
    occ.fragment(lat_in, shell)
    occ.synchronize()
    occ.removeAllDuplicates()
    occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)

    node_tags, node_xyz, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_xyz.reshape(-1, 3)
    tag2row = {int(t): i for i, t in enumerate(node_tags)}

    _, enodes = gmsh.model.mesh.getElementsByType(4)  # 4 = linear tet
    enodes = enodes.reshape(-1, 4)
    tets = np.vectorize(tag2row.get)(enodes).astype(np.int64)

    gmsh.finalize()
    return node_xyz, tets


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strut-radius", type=float, default=1/14,
                   help="circular strut radius rs (default 0.07); smaller -> "
                        "more slender -> buckles at lower strain")
    p.add_argument("--node-scale", type=float, default=2.0,
                   help="node-sphere radius / strut radius (default 2.0)")
    p.add_argument("--node-radius", type=float, default=1/3,
                   help="spherical node radius rn (overrides --node-scale); "
                        "must satisfy rs < rn < half")
    p.add_argument("--lc", type=float, default=0.1,
                   help="target element size (default 0.1)")
    p.add_argument("--order", type=int, default=1, choices=(1, 2),
                   help="geometry/element order: 1 = straight linear tets "
                        "(default, reflect/tile path), 2 = curved quadratic "
                        "tets that follow the round sphere/strut surfaces "
                        "(meshed as one periodic cell via gmsh setPeriodic, "
                        "then tiled; valid curved cells, unlike reflection)")
    p.add_argument("--half", type=float, default=0.5,
                   help="half edge of one unit cell (cell edge = 2*half, "
                        "default 0.5 -> unit cell)")
    p.add_argument("--nx", type=int, default=2, help="cells in x (default 2)")
    p.add_argument("--ny", type=int, default=2, help="cells in y (default 2)")
    p.add_argument("--nz", type=int, default=2, help="cells in z (default 2)")
    p.add_argument("--center", action="store_true",
                   help="centre the domain at the origin (RVE); default is the "
                        "DNS box [0,Nx]x[0,Ny]x[0,Nz] with the cell on [0,1]^3")
    p.add_argument("--octant-only", action="store_true",
                   help="write only the eighth (no reflection) for inspection")
    p.add_argument("--solid-skin", type=float, default=0.0, metavar="T",
                   help="DNS only: wrap the outer box in a closed solid skin of "
                        "thickness T (same units as the cell, e.g. 0.15) with the "
                        "lattice inside. 0 = off (default). Incompatible with "
                        "--center and --octant-only; full OCC build, so slower.")
    p.add_argument("--output", type=str, default=None,
                   help="output .msh path (default ./sphere_strut.msh, "
                        "./sphere_strut_NxNyNz.msh, or ./sphere_strut_octant.msh)")
    return p.parse_args()


def main():
    args = parse_args()
    rs = args.strut_radius
    rn = args.node_radius if args.node_radius is not None \
        else args.node_scale * rs
    if not 0.0 < rs < args.half:
        raise SystemExit(
            f"--strut-radius must satisfy 0 < rs < half ({args.half}); got {rs}")
    if not 0.0 < rn < args.half:
        raise SystemExit(
            f"node radius {rn:.4f} must satisfy 0 < rn < half ({args.half}) so "
            f"corner spheres stay distinct; reduce --node-scale / --node-radius")
    if rn <= rs:
        raise SystemExit(
            f"node radius {rn:.4f} should exceed strut radius {rs} so the "
            f"sphere encloses the strut junction; raise --node-scale/--node-radius")

    here = os.path.dirname(os.path.abspath(__file__))
    cell = 2.0 * args.half                            # unit-cell edge (1.0)

    skin = args.solid_skin
    if skin > 0.0:
        # --- DNS lattice wrapped in a closed solid skin (full OCC build) ----
        if args.order != 1:
            raise SystemExit(
                "--order 2 (curved tets) is only implemented for the "
                "reflect/tile path, not --solid-skin")
        if args.center:
            raise SystemExit(
                "--solid-skin is for DNS meshes and cannot be combined with "
                "--center (a skinned specimen is not a periodic RVE)")
        if args.octant_only:
            raise SystemExit("--solid-skin cannot be combined with --octant-only")
        span = min(args.nx, args.ny, args.nz) * cell
        if not 0.0 < skin < span / 2.0:
            raise SystemExit(
                f"--solid-skin thickness {skin} must satisfy "
                f"0 < T < min(W,H,D)/2 = {span / 2.0:.4f}")

        print("Building DNS lattice inside a solid skin (full OCC boolean "
              "build; this can be slow for large Nx*Ny*Nz)...")
        full_coords, full_tets = mesh_solid_skin_dns(
            args.nx, args.ny, args.nz, args.half, rs, rn, skin, args.lc)
        out = args.output or os.path.join(
            here, f"sphere_strut_{args.nx}x{args.ny}x{args.nz}_skin.msh")
        tag = (f"{args.nx}x{args.ny}x{args.nz} cells, DNS [0,N] "
               f"with solid skin T={skin}")
    elif args.octant_only:
        coords, tets = mesh_octant(args.half, rs, rn, args.lc, order=args.order)
        out = args.output or os.path.join(here, "sphere_strut_octant.msh")
        full_coords, full_tets = coords, tets
        tag = "octant"
    else:
        if args.order >= 2:
            # Curved (order-2) cells invert under reflection, so mesh the full
            # periodic cell directly with gmsh + setPeriodic instead of
            # reflecting an octant. Linear meshes keep the cheaper reflect path.
            unit_c, unit_t = mesh_full_cell_order2(args.half, rs, rn, args.lc)
        else:
            coords, tets = mesh_octant(args.half, rs, rn, args.lc, order=args.order)
            unit_c, unit_t = reflect_full(coords, tets)  # one cell, centred
        full_coords, full_tets = tile(unit_c, unit_t,
                                      args.nx, args.ny, args.nz, cell)
        if args.center:
            full_coords = full_coords - np.array(
                [args.nx, args.ny, args.nz]) * cell / 2.0
        if args.nx == args.ny == args.nz == 1:
            out = args.output or os.path.join(here, "sphere_strut.msh")
        else:
            out = args.output or os.path.join(
                here, f"sphere_strut_{args.nx}x{args.ny}x{args.nz}.msh")
        tag = (f"{args.nx}x{args.ny}x{args.nz} cells, "
               f"{'centred (RVE)' if args.center else 'DNS [0,N]'}")

    tris = boundary_triangles(full_tets)
    write_msh(out, full_coords, full_tets, tris)

    lo = full_coords.min(axis=0)
    hi = full_coords.max(axis=0)
    print(f"  mode        : {tag}")
    print(f"  strut radius: {rs}   node radius : {rn:.4f}   lc : {args.lc}")
    print(f"  element     : {'quadratic (curved)' if args.order == 2 else 'linear'}"
          f"  tet (order {args.order})")
    print(f"  nodes       : {len(full_coords)}")
    print(f"  tetrahedra  : {len(full_tets)}")
    print(f"  boundary tri: {len(tris)}")
    print(f"  bbox        : [{lo[0]:+.4f},{hi[0]:+.4f}] x "
          f"[{lo[1]:+.4f},{hi[1]:+.4f}] x [{lo[2]:+.4f},{hi[2]:+.4f}]")
    print(f"Mesh written to {out}")


if __name__ == "__main__":
    main()
