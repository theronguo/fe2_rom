import os
# nthreads = 1
# os.environ["OMP_NUM_THREADS"] = str(nthreads) 
# os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads) 
# os.environ["MKL_NUM_THREADS"] = str(nthreads)
from dolfinx import fem, io, mesh as dmesh
from ufl import grad, Identity, variable, det, tr, ln, inner
from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx.fem import petsc
import numpy as np
from slepc4py import SLEPc
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    comm = MPI.COMM_WORLD
    mesh, cell_tags, facet_tags = io.gmshio.read_from_msh("model.msh", comm, 0, gdim=3)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    space_dims = mesh.geometry.dim
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # Displacement function space
    V = fem.functionspace(mesh, ("Lagrange", 1, (space_dims, )))
    V1 = fem.functionspace(mesh, ("DG", 1, (space_dims, )))

    n_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if comm.rank == 0:
        print(f"Global DOFs: {n_dofs_global}")
        sys.stdout.flush()

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

    # --- DG0 spaces for field visualization (one value per element) ---
    TT = fem.functionspace(mesh, ("DG", 1, (space_dims, space_dims)))
    SS = fem.functionspace(mesh, ("DG", 1))

    F_func = fem.Function(TT, name="DeformationGradient")
    P_func = fem.Function(TT, name="Stress1PK")
    J_func = fem.Function(SS, name="JacobianDet")
    u_int  = fem.Function(V1, name="u")

    F_expr = fem.Expression(F, TT.element.interpolation_points())
    P_expr = fem.Expression(P, TT.element.interpolation_points())
    J_expr = fem.Expression(J, SS.element.interpolation_points())

    # Define test function
    v = ufl.TestFunction(V)

    # Define body force (e.g., gravity)
    body_force = fem.Constant(mesh, (0.0, 0.0, 0.0))

    # Residual weak form
    R = inner(grad(v), P) * dx - inner(v, body_force) * dx
    R_form = fem.form(R)

    # Jacobian of the residual
    J_nonlinear = ufl.derivative(R, u)
    J_nonlinear_form = fem.form(J_nonlinear)

    # BCs
    V_x = V.sub(0)
    V_y = V.sub(1)
    V_z = V.sub(2)
    const_0 = fem.Constant(mesh, PETSc.ScalarType(0))
    const_1 = fem.Constant(mesh, PETSc.ScalarType(0))

    # Locate boundary facets at global z_min/z_max geometrically.
    z_local = mesh.geometry.x[:, 2]
    z_min = comm.allreduce(np.min(z_local), op=MPI.MIN)
    z_max = comm.allreduce(np.max(z_local), op=MPI.MAX)
    z_tol = max(1e-8, 1e-8 * (z_max - z_min))
    fdim = mesh.topology.dim - 1

    facets_zmin = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], z_min, atol=z_tol))
    facets_zmax = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], z_max, atol=z_tol))

    facets_zmin = dmesh.locate_entities_boundary(mesh, fdim, lambda x: x[2] < 0)
    facets_zmax = dmesh.locate_entities_boundary(mesh, fdim, lambda x: x[2] > 1)

    # z = z_min fixed in all directions.
    zmin_dofs_x = fem.locate_dofs_topological(V_x, fdim, facets_zmin)
    zmin_dofs_y = fem.locate_dofs_topological(V_y, fdim, facets_zmin)
    zmin_dofs_z = fem.locate_dofs_topological(V_z, fdim, facets_zmin)
    bc_zmin_x = fem.dirichletbc(const_0, zmin_dofs_x, V_x)
    bc_zmin_y = fem.dirichletbc(const_0, zmin_dofs_y, V_y)
    bc_zmin_z = fem.dirichletbc(const_0, zmin_dofs_z, V_z)

    # z = z_max fixed in x,y and loaded in z via prescribed displacement const_1.
    zmax_dofs_x = fem.locate_dofs_topological(V_x, fdim, facets_zmax)
    zmax_dofs_y = fem.locate_dofs_topological(V_y, fdim, facets_zmax)
    zmax_dofs_z = fem.locate_dofs_topological(V_z, fdim, facets_zmax)
    bc_zmax_x = fem.dirichletbc(const_0, zmax_dofs_x, V_x)
    bc_zmax_y = fem.dirichletbc(const_0, zmax_dofs_y, V_y)
    bc_zmax_z = fem.dirichletbc(const_1, zmax_dofs_z, V_z)

    bcs = [bc_zmin_x, bc_zmin_y, bc_zmin_z, bc_zmax_x, bc_zmax_y, bc_zmax_z]

    # Reaction force via boundary traction integral: ∫ (P·n)·e_z ds over z_max.
    # Tag z_max facets so we can restrict the ds measure to that surface only.
    _zmax_sorted = np.sort(facets_zmax)
    _zmax_tags = np.full(len(_zmax_sorted), 1, dtype=np.int32)
    facet_tag_rf = dmesh.meshtags(mesh, fdim, _zmax_sorted, _zmax_tags)
    ds_zmax = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag_rf)(1)
    n_ref = ufl.FacetNormal(mesh)
    e_z = ufl.as_vector([0.0, 0.0, 1.0])
    # P·n gives the reference-config traction; project onto e_z for z-component
    reaction_form = fem.form(ufl.inner(ufl.dot(P, n_ref), e_z) * ds_zmax)

    applied_displacements = []
    reaction_forces = []

    max_iter_newton = 10
    max_iter_instab = 30
    max_amplitude = -(z_max-z_min)*0.2
    rel_tol_newton = 1e-8
    abs_tol_newton = 1e-6
    good_newton_steps = 7

    t_end = 1.0
    dt_init = 1e-1
    dt_current = dt_init
    dt_min = 1e-5
    dt_max = 1e-1
    t_current_converged = 0.0
    timestep = 0

    # Single BP file for all output fields, appended every converged step.
    vtx = io.VTXWriter(comm, "output/solution.bp", [u_int, F_func, P_func, J_func], engine="BP4")

    def write_fields(t):
        u_int.interpolate(u)
        F_func.interpolate(F_expr)
        P_func.interpolate(P_expr)
        J_func.interpolate(J_expr)
        vtx.write(t)

    # Write initial (undeformed) state
    write_fields(0.0)
    applied_displacements.append(0.0)
    reaction_forces.append(0.0)

    simulation_finished = False
    while t_current_converged < t_end:
        solver_type = PETSc.KSP.Type.CG
        trial_time = np.round(t_current_converged + dt_current, 5)
        if comm.rank == 0:
            print(f"Time: {trial_time}, dt: {dt_current}")
            sys.stdout.flush()
        const_1.value = trial_time * max_amplitude
        stable_configuration = False
        pert_amplitude = 1e1
        iter_newton = 0
        max_iter = max_iter_newton
        while not stable_configuration:
            while iter_newton < max_iter:
                is_converged = False
                residual = fem.petsc.assemble_vector(R_form)
                fem.apply_lifting(residual, [J_nonlinear_form], [bcs], x0=[u.x.petsc_vec], alpha=-1.0)
                residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem.set_bc(residual, bcs, x0=u.x.petsc_vec, scale=-1.0)

                abs_b_norm = residual.norm()
                if iter_newton == 0:
                    abs_b_norm_init = abs_b_norm
                if comm.rank == 0:
                    print(f"Newton iteration {iter_newton}, relative residual: {abs_b_norm / abs_b_norm_init}, absolute residual: {abs_b_norm}")
                    sys.stdout.flush()
                if abs_b_norm / abs_b_norm_init < rel_tol_newton or abs_b_norm < abs_tol_newton:
                    PETSc.Vec.destroy(residual)  # do not destroy matrix because needed for eigenvalue computation
                    is_converged = True
                    break

                if abs_b_norm / abs_b_norm_init > 10 or np.isnan(abs_b_norm):  # will not converge
                    PETSc.Vec.destroy(residual)
                    break

                #### assemble stiffness matrix ####
                K = fem.petsc.assemble_matrix(J_nonlinear_form, bcs=bcs)
                K.assemble()
                solver = PETSc.KSP().create(comm)
                solver.setOperators(K)
                solver.setType(solver_type)
                solver.getPC().setType(PETSc.PC.Type.GAMG)

                # solve linear system of equations
                solver.solve(-residual, du.x.petsc_vec)

                # check convergence of solver
                if solver.getConvergedReason() < 0 and solver_type == PETSc.KSP.Type.CG:
                    # solver did not converge
                    if comm.rank == 0:
                        print(f"CG solver did not converge. Reason: {solver.getConvergedReason()}. Switch to MINRES solver.")
                        sys.stdout.flush()
                    solver_type = PETSc.KSP.Type.MINRES
                    solver.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    max_iter = max_iter_instab  # allow more iterations for MINRES
                    continue
                elif solver.getConvergedReason() < 0 and solver_type == PETSc.KSP.Type.MINRES:
                    if comm.rank == 0:
                        print(f"MINRES solver did not converge. Reason: {solver.getConvergedReason()}. Reduce time step size.")
                        sys.stdout.flush()
                    solver.destroy()
                    PETSc.Vec.destroy(residual)
                    PETSc.Mat.destroy(K)
                    break
                else:
                    solver.destroy()

                # Step 1: Create full-sized du vector
                u.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
                u.x.scatter_forward()
                iter_newton += 1

                # Destroy PETSc matrix and vector
                PETSc.Vec.destroy(residual)
                PETSc.Mat.destroy(K)

            if is_converged:
                # check eigenvalues
                if comm.rank == 0:
                    print("Checking eigenvalues for stability analysis...")
                    sys.stdout.flush()
                K = fem.petsc.assemble_matrix(J_nonlinear_form, bcs=bcs)
                K.assemble()

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
                PETSc.Mat.destroy(K)

                # Get the number of converged eigenvalues
                n_conv = eigensolver.getConverged()
                eigenvalues = np.array([eigensolver.getEigenvalue(i).real for i in range(min(n_conv, 5))])
                if comm.rank == 0:
                    print("Smallest eigenvalues:", eigenvalues)
                    sys.stdout.flush()

                if np.any(eigenvalues < -1e-12):
                    target_indices = np.where(eigenvalues < 1e-12)[0]
                    eigensolver.getEigenvector(target_indices[0], eigenfunction.x.petsc_vec)
                    eigenfunction.x.scatter_forward()
                    u.x.petsc_vec.axpy(pert_amplitude, eigenfunction.x.petsc_vec)
                    u.x.scatter_forward()
                    pert_amplitude *= 2
                    if comm.rank == 0:
                        print(f"Instable solution with eigenvalue {eigenvalues[target_indices[0]]}! Perturbing solution with eigenvectors. Perturbation amplitude: {pert_amplitude}")
                        sys.stdout.flush()
                else:
                    stable_configuration = True
                    t_current_converged = trial_time
                    if iter_newton <= good_newton_steps:
                        dt_current = min(dt_current * 1.5, dt_max)
                    dt_current = min(dt_current, t_end - t_current_converged)
                    timestep += 1
                    u_last.x.array[:] = u.x.array[:]
                    u_last.x.scatter_forward()

                    # --- Write fields to BP file ---
                    write_fields(t_current_converged)

                    # --- Reaction force: ∫ (P·n)·e_z ds over z_max ---
                    rf_local = fem.assemble_scalar(reaction_form)
                    rf_total = comm.allreduce(rf_local, op=MPI.SUM)
                    applied_displacements.append(float(const_1.value))
                    reaction_forces.append(float(rf_total))
                    if comm.rank == 0:
                        print(f"Applied displacement: {const_1.value:.6f}, Reaction force z: {rf_total:.6f}")
                        sys.stdout.flush()

                # destroy eigensolver
                eigensolver.destroy()
            else:
                dt_current /= 2
                if dt_current < dt_min:
                    if comm.rank == 0:
                        print(f"Minimum time step size {dt_min} reached. Stopping simulation.")
                        sys.stdout.flush()
                    simulation_finished = True
                    break
                if comm.rank == 0:
                    print(f"Newton's method did not converge. Halving time step size to {dt_current}.")
                    sys.stdout.flush()
                u.x.array[:] = u_last.x.array[:]
                u.x.scatter_forward()
                break

        if simulation_finished:
            break

    vtx.close()

    # --- Reaction force plot (rank 0 only) ---
    if comm.rank == 0 and len(applied_displacements) > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(applied_displacements, reaction_forces, "b-o", markersize=4, linewidth=1.5)
        ax.set_xlabel("Applied displacement (m)")
        ax.set_ylabel("Reaction force z (N)")
        ax.set_title("Reaction force vs applied displacement")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/reaction_force.png", dpi=150)
        plt.close(fig)
        np.savetxt(
            "output/reaction_force.csv",
            np.column_stack([applied_displacements, reaction_forces]),
            delimiter=",",
            header="applied_displacement,reaction_force_z",
            comments="",
        )
        print("Saved output/reaction_force.png and output/reaction_force.csv")


if __name__ == "__main__":
    main()
