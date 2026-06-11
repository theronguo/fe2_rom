# First-order homogenization of a 2D RVE (FOM + ROM)

First-order computational homogenization (CH1) with periodicity on a 2D RVE, run
with both the full-order periodic solver and the POD–ECM reduced-order model.

## 1. Full-order homogenization

```bash
mpirun -np $nprocs python run_homogenization.py    # writes snapshots to output/
```

## 2. Build and run the ROM

```bash
python build_rom.py                # POD + ECM      (serial only)
python run_homogenization_rom.py   # reduced solver (serial only)
```
