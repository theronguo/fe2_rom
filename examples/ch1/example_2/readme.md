# First-order homogenization of a 3D RVE (FOM + ROM)

Uniaxial compression of a 3D periodic RVE under first-order computational
homogenization (CH1), run with both the full-order periodic solver and the
POD–ECM reduced-order model. This RVE and its ROM are reused by the FE² example
in `../example_3`.

## 1. Full-order homogenization

```bash
python create_mesh.py                              # (re)generate mesh.msh — optional
mpirun -np $nprocs python run_homogenization.py    # writes snapshots to output/
```

## 2. Build and run the ROM

```bash
python build_rom.py                # POD + ECM      (serial only)
python run_homogenization_rom.py   # reduced solver (serial only)
```
