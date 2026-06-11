# Two-scale micromorphic FE² in 2D (FOM or ROM inner)

A two-scale micromorphic problem in 2D, with a nested micromorphic RVE at every
macro quadrature point. The φ modes and ROM are reused from `../example_1`.

## Run

```bash
mpirun -np $nprocs python run_macro_micromorphic.py 0    # full RVE at each qp (FOM)
mpirun -np $nprocs python run_macro_micromorphic.py 1    # reduced RVE at each qp (ROM)
```

> Both runs need the φ modes and (for the ROM) the `ecm/` directory from
> `../example_1` — run that example first.
