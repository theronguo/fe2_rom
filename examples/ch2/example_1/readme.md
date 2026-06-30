# Second-order homogenization of a 3D RVE (FOM + ROM)

Second-order computational homogenization (CH2) on a centred 3D sphere-strut
RVE, run with both the full-order periodic solver and the POD–ECM reduced-order
model. The kinematic ansatz adds the strain-gradient enrichment

    u_total = (F̄ − I)·X + ½ X·Ḡ·X + w(X),

so the RVE is driven by the macroscopic deformation gradient `F̄` *and* its
gradient `Ḡ` (`Ḡ_iJK = ∂F̄_iJ/∂X_K`). A zero `Ḡ` recovers the first-order (ch1)
response exactly.

## 0. Generate the mesh

The RVE must be centred at the origin (the ½ X·Ḡ·X term assumes `X` symmetric
about 0):

```bash
python create_sphere_strut_mesh.py --nx 2 --ny 2 --nz 2 --center   # -> sphere_strut_2x2x2.msh
```

## 1. Full-order homogenization

```bash
mpirun -np $nprocs python run_homogenization.py    # writes snapshots to output/
```

## 2. Build and run the ROM

```bash
python build_rom.py                # POD + ECM      (serial only)
python run_homogenization_rom.py   # reduced solver (serial only)
```

## Note on training

The snapshots above are generated along a single load path with `Ḡ = 0`, so the
POD basis spans the first-order (`F̄`-driven) response. To build a ROM that is
accurate for non-zero `Ḡ`, enrich the snapshot pool with load paths that excite
`Ḡ` (rerun `run_homogenization.py` with non-zero `Gbar`, appending to `output/`)
before calling `build_rom.py`.
