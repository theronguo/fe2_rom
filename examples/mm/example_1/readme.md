# Micromorphic homogenization of a square RVE (FOM + ROM)

End-to-end micromorphic homogenization on a square (box) RVE. The pipeline first
steps up to the buckling point and runs a linear buckling analysis (LBA) to
extract the micromorphic enrichment mode φ, then solves the micromorphic RVE
problem for a prescribed loading.

## 1. Full-order model

```bash
python run_micromorphic.py    # serial only
```

Steps to the buckling point, computes φ via LBA, then runs the micromorphic
solver and the FD-verified probe.

## 2. Reduced-order model

```bash
python generate_training_data.py   # find φ, then sample 32 training points
                                   #   (serial launch; spawns parallel workers)
python build_rom.py                # POD + ECM      (serial only)
python run_micromorphic_rom.py     # reduced solver (serial only)
```
