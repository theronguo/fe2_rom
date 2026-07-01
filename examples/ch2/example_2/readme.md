# Second-order homogenization of a 2D square RVE (FOM + ROM)

Second-order computational homogenization (CH2) on the same 2D perforated square
RVE as `examples/mm/example_1` (`rve.msh`, a `[-1, 1]²` cell centred at the
origin), run with both the full-order periodic solver and the POD–ECM
reduced-order model.

The kinematic ansatz adds the strain-gradient term (paper Eq. 10)

    u_total = (F̄ − I)·X + ½ X·Ḡ·X + w(X),

so the RVE is driven by the macroscopic deformation gradient `F̄` *and* its
gradient `Ḡ` (`Ḡ_iJK = ∂F̄_iJ/∂X_K`, symmetric in `J ↔ K`). The RVE reports the
effective stress `P̄` (Eq. 26), the double stress `Q̄` (Eq. 27) and the four macro
tangents `d{P̄, Q̄}/d{F̄, Ḡ}`. The mesh
must be periodic and centred at the origin — `rve.msh` is both.

## 1. Full-order test case

```bash
conda activate fe2_rom_env
python run_homogenization.py                 # serial
mpirun -n 4 python run_homogenization.py     # or in parallel
```

Drives the RVE to a fixed `(F̄, Ḡ)` and reports/plots `P̄`, `Q̄` and the tangents
(`output/`).

## 2. Reduced-order model

```bash
python generate_training_data.py    # sample 64 (F̄, Ḡ) points → output_gen/
                                     #   (serial launch; spawns parallel workers)
python build_rom.py                 # POD + ECM → ecm/            (serial only)
python run_homogenization_rom.py    # reduced solver             (serial only)
```

`run_homogenization_rom.py` drives the reduced RVE with the *same* `(F̄, Ḡ)` as
`run_homogenization.py` and prints the relative FOM-vs-ROM error of `P̄`, `Q̄` and
the tangents.

### Couple stress in the ECM training

`build_rom.py` builds three POD bases for the empirical-cubature magic-point
selection:

  * `u_fluc` — displacement fluctuation (H¹),
  * `P`      — first Piola stress (L²),
  * `Q`      — the **double-/couple-stress density**
               `𝒴_iJK = ½(X_K P_iJ + X_J P_iK)` (L²), whose volume average is
               `Q̄`.

`𝒴` is reconstructed from
the `P` snapshots and passed to the ECM as an extra volume-integral constraint
block. Including it makes the reduced quadrature integrate the higher-order
stress accurately, not just `P̄`.

`build_rom.py` also prints the rule's coverage, e.g. `Magic points:
891 / 6396 quadrature points (13.93%) | active elements: 651 / 1066 (61.07%)`.

### Isolating POD vs ECM error

To tell whether a FOM-vs-ROM discrepancy comes from the POD (Galerkin
projection) or from the ECM (hyper-reduced cubature), set `FULL_QUADRATURE = True`
at the top of **both** `build_rom.py` and `run_homogenization_rom.py`. `build_rom.py`
then calls `ECM.use_full_quadrature()` — which keeps *every* quadrature point with
its *exact* weight (no greedy selection, `true_residual = 0`) — and writes to a
separate `ecm_full/` directory so the ECM model in `ecm/` is untouched. The
resulting ROM has **no cubature error**, so its FOM-vs-ROM error is pure POD
error; the difference from the `ecm/` run is the ECM contribution. (For this
example both give ≈3 % on `P̄` and ≈38 % on `Q̄`, so the error here is POD/branch,
not ECM.)

### Stability and buckling traversal

Both the full-order run and the training-data generation run with
`check_stability=True` and `perturb_post_buckling=True`: this RVE buckles from
~2.8 % (near-)equibiaxial compression, so on every sample the solver monitors the
reduced-Hessian on the constraint manifold and, at a bifurcation, kicks along the
buckling eigenmode to follow the physical (buckled) branch instead of the
unstable unbuckled one.
