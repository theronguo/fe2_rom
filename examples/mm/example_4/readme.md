# Micromorphic homogenization of a 2D hexagonal RVE (FOM + ROM)

Micromorphic homogenization on a polygonal (hexagonal) RVE with lattice-vector
periodicity. The buckling spectrum yields three micromorphic enrichment modes.

## 1. Mesh and full-order model

```bash
python generate_hexagonal_mesh.py    # periodic hexagonal RVE → hexagonal_rve.msh
python run_micromorphic.py           # spectrum → φ modes → micromorphic solve
```

## 2. Reduced-order model

```bash
python generate_training_data.py     # sample training data
python build_rom.py                  # POD + ECM
python run_micromorphic_rom.py       # reduced solver
```
