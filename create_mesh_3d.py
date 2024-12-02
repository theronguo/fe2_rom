import gmsh
import math

def create_mesh_with_spheres(H, L, W, num_spheres_height, num_spheres_length, num_spheres_width, sphere_radius, mesh_size):
    gmsh.initialize()
    gmsh.model.add("Box Mesh with Spheres")
    
    # Parameters for the box and spheres
    box_tag = 1
    sphere_tags = []

    # Create the box
    gmsh.model.occ.addBox(0, 0, 0, L, H, W, box_tag)

    # Calculate the spacing of the spheres
    sphere_spacing_x = L / num_spheres_length
    sphere_spacing_y = H / num_spheres_height
    sphere_spacing_z = W / num_spheres_width

    # Create the spheres
    for i in range(num_spheres_height):
        for j in range(num_spheres_length):
            for k in range(num_spheres_width):
                x_center = (j + 0.5) * sphere_spacing_x
                y_center = (i + 0.5) * sphere_spacing_y
                z_center = (k + 0.5) * sphere_spacing_z
                sphere_tag = len(sphere_tags) + 2  # Sphere tags start from 2 because box is 1
                gmsh.model.occ.addSphere(x_center, y_center, z_center, sphere_radius, sphere_tag)
                sphere_tags.append(sphere_tag)

    # Subtract spheres from the box
    new_boxes, new_boxes_map = gmsh.model.occ.fragment([(3, box_tag)] + [(3, tag) for tag in sphere_tags], [])
    new_tags = []
    for new_box in new_boxes:
        new_tags.append(new_box[1])
    gmsh.model.occ.synchronize()

    solid_tag = new_tags[-1]
    fluid_tag = new_tags[:-1]

    # Tag the fluid and solid regions
    gmsh.model.addPhysicalGroup(3, fluid_tag, 1)
    gmsh.model.setPhysicalName(3, 1, "Fluid")

    gmsh.model.addPhysicalGroup(3, [solid_tag], 2)
    gmsh.model.setPhysicalName(3, 2, "Solid")

    # Extract and tag the boundaries
    boundaries = gmsh.model.getBoundary([(3, solid_tag)], oriented=False)

    left_boundary = []
    right_boundary = []
    top_boundary = []
    bottom_boundary = []
    front_boundary = []
    back_boundary = []

    for dim, tag in boundaries:
        # Get the bounding box of the face
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        length_x = xmax - xmin
        length_y = ymax - ymin
        length_z = zmax - zmin

        if abs(xmin) < 1e-3 and length_x < 1e-3:  # Face at x = 0 (Left boundary)
            left_boundary.append(tag)
        elif abs(xmax - L) < 1e-3 and length_x < 1e-3:  # Face at x = L (Right boundary)
            right_boundary.append(tag)
        elif abs(ymin) < 1e-3 and length_y < 1e-3:  # Face at y = 0 (Bottom boundary)
            bottom_boundary.append(tag)
        elif abs(ymax - H) < 1e-3 and length_y < 1e-3:  # Face at y = H (Top boundary)
            top_boundary.append(tag)
        elif abs(zmin) < 1e-3 and length_z < 1e-3:  # Face at z = 0 (Back boundary)
            back_boundary.append(tag)
        elif abs(zmax - W) < 1e-3 and length_z < 1e-3:  # Face at z = W (Front boundary)
            front_boundary.append(tag)

    gmsh.model.addPhysicalGroup(2, left_boundary, 3)
    gmsh.model.setPhysicalName(2, 3, "Left")
    gmsh.model.addPhysicalGroup(2, right_boundary, 4)
    gmsh.model.setPhysicalName(2, 4, "Right")
    gmsh.model.addPhysicalGroup(2, top_boundary, 5)
    gmsh.model.setPhysicalName(2, 5, "Top")
    gmsh.model.addPhysicalGroup(2, bottom_boundary, 6)
    gmsh.model.setPhysicalName(2, 6, "Bottom")
    gmsh.model.addPhysicalGroup(2, front_boundary, 7)
    gmsh.model.setPhysicalName(2, 7, "Front")
    gmsh.model.addPhysicalGroup(2, back_boundary, 8)
    gmsh.model.setPhysicalName(2, 8, "Back")

    # Apply mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # Tag and assign mesh size to points
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate the mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    gmsh.write("box_with_spheres_3d_2.msh")
    gmsh.write("box_with_spheres_3d_2.vtk")

    gmsh.model.removePhysicalGroups([((3, 1))])
    gmsh.model.mesh.generate(3)
    gmsh.write("box_without_spheres_3d_2.msh")
    gmsh.write("box_without_spheres_3d_2.vtk")

    # Finalize the gmsh API
    gmsh.finalize()

# Example usage
H = 2.0  # Height of the box
L = 10.0  # Length of the box
W = 2.0  # Width of the box
num_spheres_height = 2  # Number of spheres in height
num_spheres_length = 10  # Number of spheres in length
num_spheres_width = 2  # Number of spheres in width
sphere_radius = 0.45  # Radius of each sphere
mesh_size = 0.2  # Mesh size

create_mesh_with_spheres(H, L, W, num_spheres_height, num_spheres_length, num_spheres_width, sphere_radius, mesh_size)
