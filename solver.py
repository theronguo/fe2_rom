import os
nthreads = 1
os.environ["OMP_NUM_THREADS"] = str(nthreads) 
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads) 
os.environ["MKL_NUM_THREADS"] = str(nthreads)
from dolfinx import fem, io
from ufl import grad, Identity, variable, det, tr, ln, inner
from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx.fem import petsc
import numpy as np
from slepc4py import SLEPc
import sys

comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("mesh_with_holes.msh", comm, 0, gdim=2)
space_dims = mesh.geometry.dim
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# Displacement function space
V = fem.FunctionSpace(mesh, ("Lagrange", 2, (space_dims, )))
V1 = fem.FunctionSpace(mesh, ("Lagrange", 1, (space_dims, )))

# Displacement field (Function)
u = fem.Function(V)
u_last = fem.Function(V)
du = fem.Function(V)
eigenfunction = fem.Function(V)

# Deformation gradient
F = variable(Identity(space_dims) + grad(u))

# Jacobian (determinant of F)
J = det(F)

# Right Cauchy-Green tensor
C = F.T * F

# Material parameters
mu = 1000.0  # Shear modulus
lmbda = 2000.0  # Lamé parameter

# Neo-Hookean strain energy
W = (mu / 2) * (tr(C) - 3) - mu * ln(J) + (lmbda / 2) * (ln(J))**2

# First Piola-Kirchhoff stress
P = ufl.diff(W, F)

# Define test function
v = ufl.TestFunction(V)

# Define body force (e.g., gravity)
body_force = fem.Constant(mesh, (0.0, 0.0))

# Residual weak form
R = inner(grad(v), P) * dx - inner(v, body_force) * dx
R_form = fem.form(R)

# Jacobian of the residual
J_nonlinear = ufl.derivative(R, u)
J_nonlinear_form = fem.form(J_nonlinear)

# BCs
V_x = V.sub(0)
V_y = V.sub(1)
const_0 = fem.Constant(mesh, PETSc.ScalarType(0))
const_1 = fem.Constant(mesh, PETSc.ScalarType(0))

left_dofs_x = fem.locate_dofs_topological(V_x, mesh.topology.dim - 1, facet_tags.indices[facet_tags.values == 3])
bc_left_x = fem.dirichletbc(const_0, left_dofs_x, V_x)
left_dofs_y = fem.locate_dofs_topological(V_y, mesh.topology.dim - 1, facet_tags.indices[facet_tags.values == 3])
bc_left_y = fem.dirichletbc(const_0, left_dofs_y, V_y)

right_dofs_x = fem.locate_dofs_topological(V_x, mesh.topology.dim - 1, facet_tags.indices[facet_tags.values == 4])
bc_right_x = fem.dirichletbc(const_1, right_dofs_x, V_x)
right_dofs_y = fem.locate_dofs_topological(V_y, mesh.topology.dim - 1, facet_tags.indices[facet_tags.values == 4])
bc_right_y = fem.dirichletbc(const_0, right_dofs_y, V_y)
bcs = [bc_left_x, bc_left_y, bc_right_x, bc_right_y]

max_iter_newton = 30
max_amplitude = -1
rel_tol_newton = 1e-12
abs_tol_newton = 1e-8

with io.XDMFFile(comm, f"output/solution_0.xdmf", "w") as xdmf:
    u_int = fem.Function(V1, name="u")
    u_int.interpolate(u)
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_int, 0.0)

t_end = 1.0
dt_init = 1e-1
dt_current = dt_init
t_current_converged = 0.0
timestep = 0

while t_current_converged < t_end:
    trial_time = np.round(t_current_converged + dt_current, 5)
    if comm.rank == 0:
        print(f"Time: {trial_time}")
        sys.stdout.flush()
    const_1.value = trial_time * max_amplitude
    stable_configuration = False
    pert_amplitude = 1e1
    iter_newton = 0
    while not stable_configuration:
        while iter_newton < max_iter_newton:
            is_converged = False
            residual = fem.petsc.assemble_vector(R_form)
            fem.apply_lifting(residual, [J_nonlinear_form], [bcs], x0=[u.vector], scale=-1.0)
            residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(residual, bcs, x0=u.vector, scale=-1.0)

            abs_b_norm = residual.norm()
            if iter_newton == 0:
                abs_b_norm_init = abs_b_norm
            if comm.rank == 0:
                print(f"Newton iteration {iter_newton}, relative residual: {abs_b_norm / abs_b_norm_init}, absolute residual: {abs_b_norm}")
                sys.stdout.flush()
            if abs_b_norm / abs_b_norm_init < rel_tol_newton or abs_b_norm < abs_tol_newton:
                is_converged = True
                break

            if abs_b_norm / abs_b_norm_init > 10 or np.isnan(abs_b_norm):  # will not converge
                break

            #### assemble stiffness matrix ####
            K = fem.petsc.assemble_matrix(J_nonlinear_form, bcs=bcs)
            K.assemble()
            solver = PETSc.KSP().create(comm)
            solver.setOperators(K)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.CHOLESKY)

            # solve linear system of equations
            solver.solve(-residual, du.vector)

            # Step 1: Create full-sized du vector
            u.vector.axpy(1.0, du.vector)
            u.x.scatter_forward()
            iter_newton += 1

        if is_converged:
            # check eigenvalues
            eigensolver = SLEPc.EPS().create(comm)
            eigensolver.setOperators(K)
            eigensolver.setProblemType(SLEPc.EPS.ProblemType.HEP)
            st = eigensolver.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            eigensolver.setTarget(0.0)

            # Set solver options
            eigensolver.setDimensions(nev=5)  # Compute 5 eigenvalues near the target
            eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)  # Target real eigenvalues
            eigensolver.solve()

            # Get the number of converged eigenvalues
            n_conv = eigensolver.getConverged()
            eigenvalues = np.array([eigensolver.getEigenvalue(i).real for i in range(min(n_conv, 5))])

            if np.any(eigenvalues < 1e-12):
                target_indices = np.where(eigenvalues < 1e-12)[0]
                eigensolver.getEigenvector(target_indices[0], eigenfunction.vector)
                eigenfunction.x.scatter_forward()
                u.vector.axpy(pert_amplitude, eigenfunction.vector)
                u.x.scatter_forward()
                pert_amplitude *= 2
                if comm.rank == 0:
                    print(f"Instable solution with eigenvalue {eigenvalues[target_indices[0]]}! Perturbing solution with eigenvectors. Perturbation amplitude: {pert_amplitude}")
                    sys.stdout.flush()
            else:
                stable_configuration = True
                t_current_converged = trial_time
                dt_current = min(2*dt_current, dt_init)
                dt_current = min(dt_current, t_end - t_current_converged)
                timestep += 1
                u_last.x.array[:] = u.x.array[:]
        else:
            dt_current /= 2
            if comm.rank == 0:
                print(f"Newton's method did not converge. Halving time step size to {dt_current}.")
                sys.stdout.flush()
            u.x.array[:] = u_last.x.array[:]
            break

    with io.XDMFFile(comm, f"output/solution_{timestep}.xdmf", "w") as xdmf:
        u_int = fem.Function(V1, name="u")
        u_int.interpolate(u)
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_int, t_current_converged)