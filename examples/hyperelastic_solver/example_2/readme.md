# Compression of a 3D hexagonal metamaterial

Uniaxial compression of a hexagonal lattice. Stability is monitored every Newton
step; at the bifurcation the solution is perturbed along the lowest eigenmode to
continue onto the post-buckled branch.

## Run

```bash
mpirun -np $nprocs python run_solver.py
```
