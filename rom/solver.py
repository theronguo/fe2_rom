import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # avoid OpenBLAS oversubscription
os.environ["OMP_NUM_THREADS"] = "1"      # avoid OpenMP oversubscription
os.environ["MKL_NUM_THREADS"] = "1"      # avoid MKL oversub
import numpy as np
import scipy

rom_dir = "ecm_variant2_data"
indices = np.load(os.path.join(rom_dir, "indices.npy"))
basis_u_sub = np.load(os.path.join(rom_dir, "basis_u_sub.npy"))
basis_P_sub = np.load(os.path.join(rom_dir, "basis_P_sub.npy"))
omega_sub = np.load(os.path.join(rom_dir, "omega_sub.npy"))
basis_u = np.load(os.path.join(rom_dir, "basis_u.npy"))

from hyperelastic_solver import (
    NeoHookean,
)

E = 3000.0
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

output_dir = "output"
mesh_path="holes.msh"
material = NeoHookean(mu=mu, lmbda=lmbda)

from dolfinx import io, fem, mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
comm = MPI.COMM_WORLD
gdim = 2
degree = 2
mesh, _, _ = io.gmshio.read_from_msh(mesh_path, comm, 0, gdim=gdim)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

tdim = mesh.topology.dim
submesh, cell_map, _, _ = dmesh.create_submesh(mesh, tdim, indices)
dx_sub = ufl.Measure("dx", domain=submesh)

V_sub  = fem.functionspace(submesh, ("Lagrange", degree, (gdim,)))
S_sub  = fem.functionspace(submesh, ("DG", 1, (gdim, gdim)))
Q0_sub = fem.functionspace(submesh, ("DG", 0))

V = fem.functionspace(mesh, ("Lagrange", degree, (gdim,)))
u_full = fem.Function(V)

basis_func_u_sub = fem.Function(V_sub)
basis_func_P_sub = fem.Function(S_sub)
omega_func_sub = fem.Function(Q0_sub)

omega_func_sub.x.array[:] = omega_sub

F_bar = fem.Constant(submesh, np.eye(gdim, dtype=PETSc.ScalarType))
u = fem.Function(V_sub)
F_ufl = ufl.variable(F_bar + ufl.grad(u))
P_ufl = material.first_pk_stress(F_ufl)
A_ufl = material.tangent_moduli(F_ufl)

N = basis_u_sub.shape[1]
coeffs = np.zeros(N)
u.x.array[:] = sum(coeffs[i] * basis_u_sub[:, i] for i in range(N))

v = fem.Function(V_sub)
w = fem.Function(V_sub)
r_form = fem.form(ufl.inner(P_ufl, ufl.grad(v)) * omega_func_sub * dx_sub)

i, j, k, l = ufl.indices(4)
A_grad_v = ufl.as_tensor(A_ufl[i,j,k,l] * ufl.grad(v)[k,l], (i,j))
j_form = fem.form(ufl.inner(A_grad_v, ufl.grad(w)) * omega_func_sub * dx_sub)

steps = 20
vtx = io.VTXWriter(comm, os.path.join(output_dir, "u_rom.bp"), [u_full])
vtx.write(0.0)
for l in range(steps):
    F_bar.value[0, 0] = 1 - 0.2/steps * (l+1)
    F_bar.value[1, 1] = 1 - 0.2/steps * (l+1)
    for k in range(20):
        residual = np.zeros(N)
        jacobian = np.zeros((N, N))
        for i in range(N):
            v.x.array[:] = basis_u_sub[:, i]
            R = fem.assemble_scalar(r_form)
            residual[i] = R
            for j in range(N):
                w.x.array[:] = basis_u_sub[:, j]
                J = fem.assemble_scalar(j_form)
                jacobian[i, j] = J
        print(f"Iter {k}: residual norm = {np.linalg.norm(residual)}")
        L, D, perm = scipy.linalg.ldl(jacobian)
        D[D<1e-8] = -D[D<1e-8]
        dcoeffs = np.linalg.solve(L @ D @ L.T, -residual)
        coeffs += dcoeffs
        u.x.array[:] = sum(coeffs[i] * basis_u_sub[:, i] for i in range(N))

        
    u_full.x.array[:] = sum(coeffs[i] * basis_u[:, i] for i in range(N))
    vtx.write(2*float(l+1)/steps)