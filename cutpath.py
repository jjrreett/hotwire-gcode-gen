import rich.traceback
from rich import print
import geometry
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import shapely.geometry
import shapely.plotting
import functools

DEBUG = False

geometry.DEBUG = DEBUG


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


def resample_surface(surface: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample an airfoil surface to have `num_points` evenly spaced along its arc length.

    Parameters:
        surface (np.ndarray): The original surface points (N x 2) with columns [x, z].
        num_points (int): The number of evenly spaced points.

    Returns:
        np.ndarray: The resampled surface points (num_points x 2).
    """

    if surface.shape[1] == 2:
        y = 1
    else:
        y = 2

    # Compute cumulative arc length
    distances = np.sqrt(np.diff(surface[:, 0]) ** 2 + np.diff(surface[:, y]) ** 2)
    arc_length = np.concatenate(([0], np.cumsum(distances)))

    # Create interpolation functions for x and z
    interp_x = scipy.interpolate.interp1d(arc_length, surface[:, 0], kind="linear")
    interp_z = scipy.interpolate.interp1d(arc_length, surface[:, y], kind="linear")

    # Generate new evenly spaced arc length values
    new_arc_length = np.linspace(0, arc_length[-1], num_points)

    # Compute new x and z values
    new_x = interp_x(new_arc_length)
    new_z = interp_z(new_arc_length)

    new_coords = np.column_stack((new_x, new_z))
    if y == 2:
        new_coords = np.insert(new_coords, 1, 0, axis=1)
    return new_coords


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
    coords: np.ndarray, kerf=0.1, start_point=-1, exit_point=10, plot=False
):
    coords = resample_surface(coords, 1000)
    shape = shapely.geometry.LineString(coords)
    cutline = shape.parallel_offset(kerf / 2)

    cutline_coords = np.array(cutline.coords)
    leading_edge_idx = np.argmin(cutline_coords[:, 0])

    top_cut = (
        Chainable(
            shapely.geometry.LineString(cutline_coords[: leading_edge_idx + 1][::-1])
        )
        .apply(tangent_lead_in, target_deg=50)
        .apply(
            extend_lead_out,
            lead_out_length=kerf / 2,
        )
        .apply(converge_to_x_lead_out, exit_angle_deg=30)
        .apply(connect_lead_in, shapely.geometry.Point((start_point, 0)))
        .apply(connect_lead_out, shapely.geometry.Point((exit_point, 0)))
        .result()
    )

    btm_cut = (
        Chainable(shapely.geometry.LineString(cutline_coords[leading_edge_idx:]))
        .apply(tangent_lead_in, target_deg=-45)
        .apply(
            extend_lead_out,
            lead_out_length=kerf / 2,
        )
        .apply(converge_to_x_lead_out, exit_angle_deg=-30)
        .apply(connect_lead_in, shapely.geometry.Point((start_point, 0)))
        .apply(connect_lead_out, shapely.geometry.Point((exit_point, 0)))
        .result()
    )

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            # projection="3d"
        )
        # ax.plot(shape.coords[:, 0], shape.coords[:, 1])
        shapely.plotting.plot_line(shape, add_points=False)
        shapely.plotting.plot_line(top_cut, add_points=False, color="b")
        shapely.plotting.plot_line(btm_cut, add_points=False, color="r")
        # shapely.plotting.plot_line(btm_cutline, add_points=False, color="r")
        ax.axis("equal")
        plt.show()

    return (top_cut, btm_cut)


if __name__ == "__main__":
    rich.traceback.install()
    # Example usage
    wing = geometry.get_wing_surface_from_avl_file("supergee.avl")
    compute_cutpaths(
        wing.sections[0].airfoil.xy_coords * wing.sections[0].chord, plot=True
    )
