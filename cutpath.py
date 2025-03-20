import rich.traceback
from rich import print
import geometry
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import shapely.plotting
import matplotlib.lines

DEBUG = False

geometry.DEBUG = DEBUG

inch = 25.4


class Chainable:
    def __init__(self, data):
        self.data = data

    def apply(self, func, *args, **kwargs):
        """Apply a function and return self for chaining."""
        self.data = func(self.data, *args, **kwargs)
        return self

    def result(self):
        """Return the final data."""
        return self.data


def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def extend_lead_out(linestring: shapely.geometry.LineString, lead_out_length=0.1):
    """
    Extends the last segment of the cutpath by `lead_out_length`.
    """
    coords = list(linestring.coords)
    p1, p2 = np.array(coords[-2]), np.array(coords[-1])  # Last two points
    direction = (p2 - p1) / np.linalg.norm(p2 - p1)  # Unit direction vector

    new_endpoint = p2 + direction * lead_out_length  # Extend last segment
    new_coords = coords + [tuple(new_endpoint)]

    return shapely.geometry.LineString(new_coords)


def tangent_lead_in(linestring: shapely.geometry.LineString, target_deg=60):
    """
    Returns a new LineString where the first segment has a tangent angle closest to target_deg
    with respect to the x-axis. The new LineString starts from this point onward.
    """
    angles = []
    coords = list(linestring.coords)

    # Compute first segment tangent
    first_tangent = np.array(coords[1]) - np.array(coords[0])
    angles.append(np.atan2(first_tangent[1], first_tangent[0]))

    # Compute tangents for other segments
    for i, (a, b, c) in enumerate(zip(coords, coords[1:], coords[2:])):
        a, b, c = [np.array(pt) for pt in (a, b, c)]
        tangent = c - a
        theta = np.atan2(tangent[1], tangent[0])
        angles.append(theta)

    # Convert target angle to radians
    target_rad = np.deg2rad(target_deg)

    # Find index of first segment whose tangent angle crosses target_deg
    diff_angles = np.abs(np.array(angles) - target_rad)
    tangent_index = np.argmin(diff_angles)  # Find the closest match

    px, py = coords[tangent_index]
    theta = angles[tangent_index - 1]  # Adjust index since angles list is shorter

    # Compute intersection with the x-axis (y=0)
    if theta == 0:  # Avoid division by zero
        x_intersect = px
    else:
        x_intersect = px - (py / np.tan(theta))

    intersection_point = (x_intersect, 0)

    # Create a new LineString starting from the intersection point
    new_coords = [intersection_point] + coords[tangent_index:]

    return shapely.geometry.LineString(new_coords)


def indexed_lead_in(
    linestring: shapely.geometry.LineString, index, x_intersect, y_intersect
):
    coords = list(linestring.coords)
    new_coords = [(x_intersect, y_intersect)] + coords[index:]
    return shapely.geometry.LineString(new_coords)


def converge_to_x_lead_out(linestring: shapely.geometry.LineString, exit_angle_deg=30):
    """
    Adds a lead-out segment from the last point of the LineString to the x-axis at a given angle.

    Parameters:
    - linestring (LineString): The original cutpath.
    - exit_angle_deg (float): The angle (in degrees) at which the lead-out should intersect the x-axis.

    Returns:
    - LineString: The new cutpath including the lead-out.
    """
    coords = list(linestring.coords)
    last_point = np.array(coords[-1])  # Last point of the cutpath

    # Convert exit angle to radians
    exit_angle_rad = np.deg2rad(exit_angle_deg)

    # Compute intersection with the x-axis (y = 0)
    if np.tan(exit_angle_rad) == 0:  # Avoid division by zero
        lead_out_x = last_point[0]  # Vertical drop
    else:
        lead_out_x = last_point[0] + (last_point[1] / np.tan(exit_angle_rad))

    lead_out_point = (lead_out_x, 0)

    # Create new LineString with lead-out segment
    new_coords = coords + [lead_out_point]
    return shapely.geometry.LineString(new_coords)


def connect_lead_in(linestring: shapely.geometry.LineString, point):
    """
    Adds a line segment connecting a specified reference point to the start of the lead-in
    and the end of the lead-out.

    Parameters:
    - linestring (LineString): The original path with lead-in and lead-out.
    - ref_point (tuple): The (x, y) coordinate to connect to.

    Returns:
    - LineString: The new cutpath with connection segments.
    """
    coords = list(linestring.coords)
    new_coords = [point] + coords
    return shapely.geometry.LineString(new_coords)


def connect_lead_out(linestring: shapely.geometry.LineString, point):
    """
    Adds a line segment connecting a specified reference point to the start of the lead-in
    and the end of the lead-out.

    Parameters:
    - linestring (LineString): The original path with lead-in and lead-out.
    - ref_point (tuple): The (x, y) coordinate to connect to.

    Returns:
    - LineString: The new cutpath with connection segments.
    """
    coords = list(linestring.coords)
    new_coords = coords + [point]
    return shapely.geometry.LineString(new_coords)


def compute_cutpaths(
    coords: np.ndarray,
    kerf=0.1,
    start_point=-1,
    exit_point=10,
):
    shape = shapely.geometry.LineString(coords)
    shape = shapely.geometry.LineString(
        shape.interpolate(np.linspace(0, 1, 1000, endpoint=True), normalized=True)
    )
    cutline = shape.parallel_offset(kerf / 2)

    cutline_coords = np.array(cutline.coords)
    leading_edge_idx = np.argmin(cutline_coords[:, 0])

    leading_edge_y = cutline_coords[leading_edge_idx, 1]

    top_cut = (
        Chainable(
            shapely.geometry.LineString(cutline_coords[: leading_edge_idx + 1][::-1])
        )
        .apply(
            shapely.geometry.LineString.interpolate,
            distance=np.linspace(0, 1, 1000, endpoint=True),
            normalized=True,
        )
        .apply(shapely.geometry.LineString)
        .apply(
            indexed_lead_in,
            25,
            cutline_coords[leading_edge_idx, 0] - kerf,
            leading_edge_y,
        )
        .apply(
            extend_lead_out,
            lead_out_length=kerf / 2 + 0.5,
        )
        .apply(connect_lead_in, shapely.geometry.Point((start_point, leading_edge_y)))
        .result()
    )

    btm_cut = (
        Chainable(shapely.geometry.LineString(cutline_coords[leading_edge_idx:]))
        .apply(
            shapely.geometry.LineString.interpolate,
            distance=np.linspace(0, 1, 1000, endpoint=True),
            normalized=True,
        )
        .apply(shapely.geometry.LineString)
        .apply(
            indexed_lead_in,
            20,
            cutline_coords[leading_edge_idx, 0] - kerf,
            leading_edge_y,
        )
        .apply(
            extend_lead_out,
            lead_out_length=kerf / 2 + 0.5,
        )
        .apply(connect_lead_in, shapely.geometry.Point((start_point, leading_edge_y)))
        .result()
    )

    return (top_cut, btm_cut)


class LineDataUnits(matplotlib.lines.Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72.0 / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def find_intersection(x1, y1, z1, x2, y2, z2, target_z):
    """
    Find the intersection of the line formed by point1 and point2 with a plane at target_z.

    Parameters:
        point1 (tuple): The first point (x, y, z).
        point2 (tuple): The second point (x, y, z).
        target_z (float): The target z-value of the plane.

    Returns:
        tuple: The intersection point (x, y, target_z).
    """

    # If the line is vertical (z1 == z2), no intersection is needed; just return the target_z
    if z1 == z2:
        return (x1, y1, target_z)

    # Compute the parameter t for interpolation at target_z
    t = (target_z - z1) / (z2 - z1)

    # Interpolate x and y at target_z
    x_intersection = x1 + t * (x2 - x1)
    y_intersection = y1 + t * (y2 - y1)

    return (x_intersection, y_intersection, target_z)


def project_cut_paths_to_planes(
    left_cut: shapely.geometry.LineString,
    right_cut: shapely.geometry.LineString,
    z_left,
    z_right,
    z_project_left,
    z_project_right,
):
    """
    Project the left and right cut paths onto a parallel plane at z_plane.

    Parameters:
        left_cut (LineString): The left cut path as a LineString.
        right_cut (LineString): The right cut path as a LineString.
        z_plane (float): The z-coordinate of the plane.

    Returns:
        tuple: Two LineStrings representing the projected left and right cut paths.
    """
    left_projected = []
    right_projected = []

    # Iterate over corresponding points in both cut paths
    for left_point, right_point in zip(left_cut.coords, right_cut.coords):
        # Project the left and right cut points to the target plane
        projected_left_point = find_intersection(
            *left_point, z_left, *right_point, z_right, z_project_left
        )
        projected_right_point = find_intersection(
            *left_point, z_left, *right_point, z_right, z_project_right
        )

        # Append the projected points
        left_projected.append(projected_left_point)
        right_projected.append(projected_right_point)

    # Convert projected points back into LineStrings
    left_proj_line = shapely.geometry.LineString(left_projected)
    right_proj_line = shapely.geometry.LineString(right_projected)

    return left_proj_line, right_proj_line


def compute_gcode(
    left_top_cut: shapely.geometry.LineString,
    right_top_cut: shapely.geometry.LineString,
    left_btm_cut: shapely.geometry.LineString,
    right_btm_cut: shapely.geometry.LineString,
):
    gcode = """\
(Program Start)
G17 G21 G90 G40 G49 G64
(Initial Height)
"""
    assert len(left_top_cut.coords) == len(right_top_cut.coords)

    for (x, y, _), (u, z, _) in zip(left_top_cut.coords, right_top_cut.coords):
        gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f}\n"

    gcode += "(TODO transition between top and bottom cut)"

    assert len(left_btm_cut.coords) == len(right_btm_cut.coords)
    for (x, y, _), (u, z, _) in zip(left_btm_cut.coords, right_btm_cut.coords):
        gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f}\n"

    gcode += "M2\n"

    return gcode


def plot_line_3d(line: shapely.geometry.LineString, ax=None, *args, **kwargs):
    if not ax:
        ax = plt.gca()

    c = np.array(line.coords)
    ax.plot(c[:, 0], c[:, 1], c[:, 2], *args, **kwargs)


if __name__ == "__main__":
    rich.traceback.install()
    # Example usage
    wing = geometry.get_wing_surface_from_avl_file("synergyII/synergyII.avl")

    for section in wing.sections:
        print(section.airfoil_file)
        if section is None:
            raise ValueError
    # print(wing)
    from main import plot_3d
    # plot_3d(wing)

    import pyvista as pv
    import numpy as np

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Define Machine Volume (cutting area)
    machine_bounds = pv.Box(bounds=(0, 20, 0, 10, 0, 20))
    plotter.add_mesh(machine_bounds, color="gray", opacity=0.3, style="wireframe")

    # Define Stock Volume (material to be cut)
    stock = pv.Box(bounds=(0, 10, 0, 2, 3, 11))
    plotter.add_mesh(stock, color="pink", opacity=0.5)

    left = wing.sections[3]
    right = wing.sections[4]
    offset = np.array([[left.x_le, 0]])
    # fig = plt.figure()
    # ax = fig.add_subplot()

    left_coords = left.get_transformed_xy_coords()
    pv_line = pv.MultipleLines(
        np.column_stack((left_coords, np.full(left_coords.shape[0], 3)))
    )
    plotter.add_mesh(pv_line, color="blue", line_width=2)

    # plot_line_3d(
    #     shapely.geometry.LineString(
    #         np.column_stack((left_coords, np.full(left_coords.shape[0], 3)))
    #     ),
    #     ax,
    # )

    left_top_cut, left_btm_cut = compute_cutpaths(
        left_coords,
        # plot=ax,
    )
    right_coords = right.get_transformed_xy_coords()
    # plot_line_3d(
    #     shapely.geometry.LineString(
    #         np.column_stack(
    #             (
    #                 right_coords,
    #                 np.full(right_coords.shape[0], (right.y_le + 3 - left.y_le)),
    #             )
    #         )
    #     ),
    #     ax,
    # )
    right_top_cut, right_btm_cut = compute_cutpaths(
        right_coords,
        # plot=ax,
        #  color="r"
    )

    left_top_cut, right_top_cut = project_cut_paths_to_planes(
        left_top_cut, right_top_cut, 3, right.y_le + 3 - left.y_le, 0, 15
    )
    left_btm_cut, right_btm_cut = project_cut_paths_to_planes(
        left_btm_cut, right_btm_cut, 3, right.y_le + 3 - left.y_le, 0, 15
    )

    # plot_line_3d(left_top_cut, ax)
    # plot_line_3d(right_top_cut, ax)
    # plot_line_3d(left_btm_cut, ax)
    # plot_line_3d(right_btm_cut, ax)

    # shapely.plotting.plot_line(left_top_cut, add_points=False)
    # shapely.plotting.plot_line(left_btm_cut, add_points=False)
    # shapely.plotting.plot_line(right_top_cut, add_points=False, color="r")
    # shapely.plotting.plot_line(right_btm_cut, add_points=False, color="r")

    # ax.axis("equal")
    # plt.show()
    plotter.show()

    gcode = compute_gcode(
        left_top_cut,
        right_top_cut,
        left_btm_cut,
        right_btm_cut,
    )
    import pathlib

    pathlib.Path("out.nc").write_text(gcode)
