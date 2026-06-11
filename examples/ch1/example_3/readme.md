# Two-scale FE² on a single macro element (FOM or ROM inner)

A two-scale problem on a macroscopic unit cube meshed with a single hexahedral
element (8 quadrature points), with a nested first-order 3D RVE at every
quadrature point. The RVE mesh and ROM are reused from `../example_2`.

## Run

```bash
python run_macro.py 1    # full periodic RVE at each qp (FOM — slower)
python run_macro.py 0    # POD–ECM reduced RVE at each qp (ROM)
```

> The ROM run (`0`) reuses the ROM trained in `../example_2` — build it there
> first (`python build_rom.py`).
