import gmsh
import math

def create_mesh_with_holes(H, L, num_holes_height, num_holes_length, hole_radius, mesh_size):
    gmsh.initialize()
    gmsh.model.add("Rectangular Mesh with Holes")
    
    # Parameters for the rectangle and holes
    rect_tag = 1
    hole_tags = []

    # Create the rectangle
    gmsh.model.occ.addRectangle(0, 0, 0, L, H, rect_tag)

    # Calculate the spacing of the holes
    hole_spacing_x = L / num_holes_length
    hole_spacing_y = H / num_holes_height

    # Create the holes
    for i in range(num_holes_height):
        for j in range(num_holes_length):
            x_center = (j + 0.5) * hole_spacing_x
            y_center = (i + 0.5) * hole_spacing_y
            hole_tag = len(hole_tags) + 1 + 1  # Hole tags start from 2 because rectangle is 1
            gmsh.model.occ.addDisk(x_center, y_center, 0, hole_radius, hole_radius, hole_tag)
            hole_tags.append(hole_tag)

    new_boxes, new_boxes_map = gmsh.model.occ.fragment([(2, box) for box in [rect_tag]+hole_tags], [])
    new_tags = []
    for new_box in new_boxes:
        new_tags.append(new_box[1])
    gmsh.model.occ.synchronize()

    solid_tag = new_tags[-1]
    fluid_tag = new_tags[:-1]

    # Tag the fluid and solid regions
    gmsh.model.addPhysicalGroup(2, fluid_tag, 1)
    gmsh.model.setPhysicalName(2, 1, "Fluid")

    gmsh.model.addPhysicalGroup(2, [solid_tag], 2)
    gmsh.model.setPhysicalName(2, 2, "Solid")

    # Extract and tag the boundaries
    boundaries = gmsh.model.getBoundary([(2, solid_tag)], oriented=False)

    left_boundary = []
    right_boundary = []
    top_boundary = []
    bottom_boundary = []

    for dim, tag in boundaries:
        # Get the bounding box of the edge
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        length_x = xmax - xmin
        length_y = ymax - ymin

        if abs(xmin) < 1e-3 and length_x < 1e-3:  # Vertical line at x = 0 (Left boundary)
            left_boundary.append(tag)
        elif abs(xmax - L) < 1e-3 and length_x < 1e-3:  # Vertical line at x = rect_length (Right boundary)
            right_boundary.append(tag)
        elif abs(ymin) < 1e-3 and length_y < 1e-3:  # Horizontal line at y = 0 (Bottom boundary)
            bottom_boundary.append(tag)
        elif abs(ymax - H) < 1e-3 and length_y < 1e-3:  # Horizontal line at y = rect_width (Top boundary)
            top_boundary.append(tag)

    gmsh.model.addPhysicalGroup(1, left_boundary, 3)
    gmsh.model.setPhysicalName(1, 3, "Left")
    gmsh.model.addPhysicalGroup(1, right_boundary, 4)
    gmsh.model.setPhysicalName(1, 4, "Right")
    gmsh.model.addPhysicalGroup(1, top_boundary, 5)
    gmsh.model.setPhysicalName(1, 5, "Top")
    gmsh.model.addPhysicalGroup(1, bottom_boundary, 6)
    gmsh.model.setPhysicalName(1, 6, "Bottom")

    # Synchronize the CAD model
    gmsh.model.occ.synchronize()

    # Apply mesh size
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # Tag and assign mesh size to points
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    gmsh.write("mesh_without_holes.msh")

    gmsh.model.removePhysicalGroups([((2, 1))])
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh_with_holes.msh")

    # Finalize the gmsh API
    gmsh.finalize()

# Example usage
H = 10.0  # Height of the rectangle
L = 10.0  # Length of the rectangle
num_holes_height = 10  # Number of holes in height
num_holes_length = 10  # Number of holes in length
hole_radius = 0.46  # Radius of each hole
mesh_size = 0.05  # Mesh size

create_mesh_with_holes(H, L, num_holes_height, num_holes_length, hole_radius, mesh_size)
