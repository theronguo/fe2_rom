# Local vs. global buckling — van Bree et al. (2020), §4.1

Two-scale validation of the micromorphic FE² scheme against a fully-resolved
DNS, for a perforated elastomeric column that switches between **global**
(Euler-type) and **local** (patterning) buckling depending on its slenderness.

The same square RVE (2ℓ × 2ℓ, four circular holes, P2 triangles, hₘ = ℓ/10)
is driven by four solvers, each producing a nominal-stress curve
**P₂₂ vs. applied strain u/H**:

| Tag        | Script                  | Description                                             |
| ---------- | ----------------------- | ------------------------------------------------------- |
| **DNS**    | `run_macro_dns.py`      | fully-resolved reference                                |
| **MM**     | `run_macro_mm.py`       | micromorphic FE² (one patterning enrichment mode φ₁)    |
| **CH1**    | `run_macro_ch1.py`      | classical first-order FE² — baseline, deviates from DNS |

## Conventions

- Activate the environment first: `conda activate fe2_rom_env`.
- `$nprocs` is the number of MPI ranks.
- `--hM` is the macro element size in units of ℓ (**one value per run**). The
  loops below sweep `{1, 2, 4, 8, 15, 30}`; each run writes its own
  `output_hM<h>/`.

## 1. DNS (reference)

```bash
python create_dns_mesh.py --nx 6 --ny 30        # → dns_6x30.msh
mpirun -np $nprocs python run_macro_dns.py    # → output_dns_6x30/
```

## 2. Micromorphic FE² (MM)

```bash
# RVE mesh (centred at the origin, as the micromorphic ansatz requires)
python create_dns_mesh.py --nx 2 --ny 2 --center --output rve.msh   # → rve.msh

for h in 1 2 4 8 15 30; do
  mpirun -np $nprocs python run_macro_mm.py --hM $h               # → output_hM$h/
done
```

> For the coarse `hM` values the macro mesh may have too few elements to split
> over `$nprocs` ranks — drop the rank count for those runs.
>
> Add `--objective` to use the co-rotational objectivity reduction (`F̄ = R U`,
> `φ → R φ`) in each RVE — same result, fewer adjoint solves.

## 3. First-order FE² (CH1, baseline)

```bash
python create_dns_mesh.py --nx 2 --ny 2 --center --output rve.msh   # if not already built

for h in 1 2 4 8 15 30; do
  mpirun -np $nprocs python run_macro_ch1.py --hM $h              # → output_ch1_hM$h/
done
```

Included for comparison only: classical first-order homogenization does not manage to traverse through the first buckling point.
