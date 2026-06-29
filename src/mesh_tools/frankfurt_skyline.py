from pathlib import Path
import gmsh

'''
ex05 (a)
'''

def add_polyline(points, mesh_size):
    """
    Add a connected polyline.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Ordered (x, y) coordinates.
    mesh_size : float
        Characteristic mesh size at the points.

    Returns
    -------
    point_tags : list[int]
        Tags of the created Gmsh points.
    line_tags : list[int]
        Tags of the created Gmsh lines.
    """
    point_tags = [
        gmsh.model.geo.addPoint(x, y, 0.0, mesh_size)
        for x, y in points
    ]

    line_tags = [
        gmsh.model.geo.addLine(point_tags[i], point_tags[i + 1])
        for i in range(len(point_tags) - 1)
    ]

    return point_tags, line_tags


def add_physical_group(dimension, entities, name):
    """Create and name a Gmsh physical group."""
    group_tag = gmsh.model.addPhysicalGroup(dimension, entities)
    gmsh.model.setPhysicalName(dimension, group_tag, name)
    return group_tag


def write_skyline(width=40.0, height=30.0, h_min=0.1, h_max=0.5, output_file="frankfurt_skyline.msh"):
    gmsh.initialize()
    gmsh.model.add("frankfurt_skyline")

    # ------------------------------------------------------------
    # Outer-domain corner points
    #
    # Boundary traversal:
    #
    # bottom-left -> top-left -> top-right -> bottom-right
    # -> skyline from right to left -> bottom-left
    # ------------------------------------------------------------
    bottom_left = gmsh.model.geo.addPoint(
        0.0, 0.0, 0.0, h_max
    )
    top_left = gmsh.model.geo.addPoint(
        0.0, height, 0.0, h_max
    )
    top_right = gmsh.model.geo.addPoint(
        width, height, 0.0, h_max
    )
    bottom_right = gmsh.model.geo.addPoint(
        width, 0.0, 0.0, h_max
    )

    left_side = gmsh.model.geo.addLine(bottom_left, top_left)
    top_edge = gmsh.model.geo.addLine(top_left, top_right)
    right_side = gmsh.model.geo.addLine(top_right, bottom_right)

    # ------------------------------------------------------------
    # Simplified Frankfurt skyline
    #
    # Coordinates are ordered from right to left because the outer
    # boundary is traversed clockwise.
    #
    # Recognizable simplified features:
    # - low riverside buildings
    # - stepped modern tower
    # - Main Tower-like antenna
    # - church / cathedral spire
    # - Messeturm-like pyramid roof
    # ------------------------------------------------------------
    max_y = 15.0

    skyline_points = [
        # Right ground
        (40.0, 0.0),
        (37.0, 0.0),

        # Low building
        (37.0, 3.0),
        (34.8, 3.0),
        (34.8, 4.2),
        (32.5, 4.2),
        (32.5, 0.0),

        # Stepped skyscraper
        (30.5, 0.0),
        (30.5, 7.0),
        (29.7, 7.0),
        (29.7, 9.0),
        (28.9, 9.0),
        (28.9, 11.0),
        (27.5, 11.0),
        (27.5, 0.0),

        # Main Tower-like building with antenna
        (25.0, 0.0),
        (25.0, 10.5),
        (24.5, 10.5),
        (24.5, 13.2),
        (24.0, 13.2),
        (23.5, max_y),   # antenna top
        (23.0, 13.2),
        (22.5, 13.2),
        (22.5, 10.5),
        (22.0, 10.5),
        (22.0, 0.0),

        # Smaller buildings
        (20.0, 0.0),
        (20.0, 5.5),
        (18.0, 5.5),
        (18.0, 0.0),

        # Church body and tower
        (16.7, 0.0),
        (16.7, 4.5),
        (15.7, 4.5),
        (15.7, 8.2),
        (14.8, 10.0),   # church spire
        (13.9, 8.2),
        (13.9, 4.5),
        (12.8, 4.5),
        (12.8, 0.0),

        # Messeturm-like building with pyramid roof
        (10.8, 0.0),
        (10.8, 8.0),
        (9.7, 10.5),
        (8.6, 8.0),
        (8.6, 0.0),

        # Low historic buildings
        (7.0, 0.0),
        (7.0, 3.2),
        (5.8, 4.0),
        (4.6, 3.2),
        (4.6, 0.0),

        # Left ground
        (0.0, 0.0),
    ]

    # The first and last skyline points already exist as the
    # bottom-right and bottom-left outer corners. Create only the
    # intermediate skyline points.
    skyline_point_tags = [bottom_right]

    for x, y in skyline_points[1:-1]:
        # decrease h linearly with height
        h = (y / max_y) * (h_min - h_max) + h_max
        tag = gmsh.model.geo.addPoint(
            width * x / 40.0, height * y / 30.0, 0.0, h
        )
        skyline_point_tags.append(tag)

    skyline_point_tags.append(bottom_left)

    skyline_lines = [
        gmsh.model.geo.addLine(
            skyline_point_tags[i],
            skyline_point_tags[i + 1]
        )
        for i in range(len(skyline_point_tags) - 1)
    ]

    # ------------------------------------------------------------
    # Create the closed domain
    # ------------------------------------------------------------
    boundary_lines = [
        left_side,
        top_edge,
        right_side,
        *skyline_lines,
    ]

    curve_loop = gmsh.model.geo.addCurveLoop(boundary_lines)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Transfer the geometry into the Gmsh model before creating
    # physical groups or generating the mesh.
    gmsh.model.geo.synchronize()

    # ------------------------------------------------------------
    # Physical groups for boundary conditions
    # ------------------------------------------------------------
    add_physical_group(
        dimension=1,
        entities=[top_edge, *skyline_lines],
        name="DirichletRand",
    )

    add_physical_group(
        dimension=1,
        entities=[left_side, right_side],
        name="NeumannRand",
    )

    add_physical_group(
        dimension=2,
        entities=[surface],
        name="FrankfurtSkyline",
    )

    # Generate a triangular two-dimensional mesh.
    gmsh.model.mesh.generate(2)

    # Create parent directories if they don't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    gmsh.write(output_file)
    gmsh.finalize()

    return output_file
