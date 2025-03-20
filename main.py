from dataclasses import dataclass
from xml.etree.ElementTree import TreeBuilder
import rich.traceback
from rich import print
import geometry
import numpy as np
import matplotlib.pyplot as plt
import cutpath
import tomllib
import pathlib
import pyvista as pv
import shapely.geometry
import shapely
import shapely.affinity


def plot_3d(wing: geometry.Surface):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for section in wing.sections:
        print(section.airfoil.name, section.airfoil.xz_coords.shape)

        coords = geometry.Airfoil.get_transformed_coords(
            section.airfoil.xz_coords,
            section.chord,
            section.angle,
            section.get_transform(),
        )

        ax.plot(
            *geometry.Airfoil.get_plot_3d_args(coords),
            label=f"{section.airfoil.name} at Y={section.y_le}",
        )

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Wing Sections")
    ax.axis("equal")

    plt.legend()
    plt.show()


def plot_2d(wing: geometry.Surface):
    fig = plt.figure()
    ax = fig.add_subplot(
        111,
        # projection="3d"
    )

    for section in wing.sections:
        print(section.airfoil.name, section.airfoil.xz_coords.shape)

        coords = geometry.Airfoil.get_transformed_coords(
            section.airfoil.xz_coords,
            section.chord,
            section.angle,
            section.get_transform(),
        )

        ax.plot(
            *geometry.Airfoil.get_plot_2d_args(coords),
            label=f"{section.airfoil.name} at Y={section.y_le}",
        )

    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    ax.set_title("Wing Sections")
    ax.axis("equal")

    plt.legend()
    plt.show()


def setup_scene(plotter: pv.Plotter, machine_def):
    plotter.show_axes()
    # plotter.set_viewup([0, 1, 0])
    plotter.camera_position = [
        # Camera position
        (
            0.8 * -machine_def["work"]["volume"]["deltaX"],
            2 * machine_def["work"]["volume"]["deltaY"],
            machine_def["work"]["volume"]["deltaZ"],
        ),
        # Focal point (center of scene)
        (
            machine_def["work"]["volume"]["deltaX"] / 2,
            machine_def["work"]["volume"]["deltaY"] / 2,
            -machine_def["work"]["volume"]["deltaZ"] / 2,
        ),
        # View-up vector
        (0, 1, 0),
    ]

    # Define Machine Volume (cutting area)
    machine_bounds = pv.Box(
        bounds=(
            0,
            machine_def["work"]["volume"]["deltaX"],
            0,
            machine_def["work"]["volume"]["deltaY"],
            0,
            -machine_def["work"]["volume"]["deltaZ"],
        )
    )
    plotter.add_mesh(machine_bounds, color="gray", opacity=0.3, style="wireframe")

    if "stock" not in machine_def:
        return plotter
    stock = pv.Box(
        bounds=(
            machine_def["stock"]["offset"]["absoluteX"],
            machine_def["stock"]["offset"]["absoluteX"]
            + machine_def["stock"]["volume"]["deltaX"],
            machine_def["stock"]["offset"]["absoluteY"],
            machine_def["stock"]["offset"]["absoluteY"]
            + machine_def["stock"]["volume"]["deltaY"],
            machine_def["stock"]["offset"]["absoluteZ"],
            machine_def["stock"]["offset"]["absoluteZ"]
            - machine_def["stock"]["volume"]["deltaZ"],
        )
    )
    plotter.add_mesh(stock, color="pink", opacity=0.3)
    return plotter


@dataclass
class WingSection:
    inner_airfoil: shapely.geometry.LineString
    outer_airfoil: shapely.geometry.LineString
    deltaZ: float
    alpha_inner: float
    alpha_outer: float
    inner_le_x: float
    inner_le_y: float
    outer_le_x: float
    outer_le_y: float
    inner_chord: float
    outer_chord: float

    _transformed_inner_airfoil: shapely.geometry.LineString = None
    _transformed_outer_airfoil: shapely.geometry.LineString = None

    def __post_init__(self):
        print(f"transforming inner airfoil by {self.inner_le_x=}")
        self._transformed_inner_airfoil = self.transform_airfoil(
            self.inner_airfoil,
            self.inner_chord,
            self.alpha_inner,
            self.inner_le_x,
            self.inner_le_y,
        )

        # Apply transformations to the outer airfoil
        print(f"transforming outer airfoil by {self.outer_le_x=}")
        self._transformed_outer_airfoil = self.transform_airfoil(
            self.outer_airfoil,
            self.outer_chord,
            self.alpha_outer,
            self.outer_le_x,
            self.outer_le_y,
        )

    @staticmethod
    def transform_airfoil(
        airfoil: shapely.geometry.LineString,
        chord: float,
        alpha: float,
        le_x: float,
        le_y: float,
    ):
        """
        Transforms an airfoil shape by:
        1. Rotating by `alpha` degrees around -Z (clockwise)
        2. Scaling by the `chord`
        3. Translating to the leading edge position `(le_x, le_y)`

        Args:
            airfoil (LineString): The original airfoil shape
            chord (float): Chord length for scaling
            alpha (float): Rotation angle in degrees (clockwise)
            le_x (float): Leading edge x-position
            le_y (float): Leading edge y-position

        Returns:
            LineString: The transformed airfoil shape
        """
        airfoil = shapely.affinity.rotate(
            airfoil, -alpha, origin=(0, 0), use_radians=False
        )  # Rotate clockwise
        airfoil = shapely.affinity.scale(
            airfoil, xfact=chord, yfact=chord, origin=(0, 0)
        )  # Scale by chord
        airfoil = shapely.affinity.translate(
            airfoil,
            xoff=le_x,  # yoff=le_y
        )  # Move to leading edge
        return airfoil


def cut_section(section, machine_def):
    leading_edge_offset_x = machine_def["stock"]["offset"]["absoluteX"] + 0.25
    leading_edge_offset_y = (
        machine_def["stock"]["offset"]["absoluteY"]
        + machine_def["stock"]["volume"]["deltaY"] / 2
    )
    leading_edge_offset_z = machine_def["stock"]["offset"]["absoluteZ"]

    plotter = pv.Plotter()
    setup_scene(plotter, machine_def)

    points = shapely.get_coordinates(section._transformed_inner_airfoil, include_z=True)
    points[:, 0] += leading_edge_offset_x
    points[:, 1] += leading_edge_offset_y
    points[:, 2] = leading_edge_offset_z

    inner_top_cut, inner_btm_cut = cutpath.compute_cutpaths(
        points,
        kerf=machine_def["cut"]["kerf"],
        start_point=machine_def["stock"]["offset"]["absoluteX"] - 1,
    )
    l = pv.lines_from_points(points)
    plotter.add_mesh(l, color="#fc33ff")

    points = shapely.get_coordinates(section._transformed_outer_airfoil, include_z=True)
    points[:, 0] += leading_edge_offset_x
    points[:, 1] += leading_edge_offset_y
    points[:, 2] = leading_edge_offset_z - section.deltaZ
    outer_top_cut, outer_btm_cut = cutpath.compute_cutpaths(
        points,
        kerf=machine_def["cut"]["kerf"],
        start_point=machine_def["stock"]["offset"]["absoluteX"] - 1,
    )
    l = pv.lines_from_points(
        points,
    )
    plotter.add_mesh(l, color="b")

    inner_top_code, outer_top_code = cutpath.project_cut_paths_to_planes(
        inner_top_cut,
        outer_top_cut,
        leading_edge_offset_z,
        leading_edge_offset_z - section.deltaZ,
        0,
        -machine_def["work"]["volume"]["deltaZ"],
    )
    left_btm_code, right_btm_code = cutpath.project_cut_paths_to_planes(
        inner_btm_cut,
        outer_btm_cut,
        leading_edge_offset_z,
        leading_edge_offset_z - section.deltaZ,
        0,
        -machine_def["work"]["volume"]["deltaZ"],
    )

    for linestring in [inner_top_code, outer_top_code, left_btm_code, right_btm_code]:
        l = pv.lines_from_points(
            shapely.get_coordinates(linestring, include_z=True),
        )
        plotter.add_mesh(l, color="g")
    plotter.show()


if __name__ == "__main__":
    rich.traceback.install()

    # Example usage
    wing = geometry.get_wing_surface_from_avl_file("synergyII/synergyII.avl")

    idx = 3
    deltaZ = wing.sections[idx + 1].y_le - wing.sections[idx].y_le

    section = WingSection(
        inner_airfoil=shapely.geometry.LineString(wing.sections[idx].airfoil.xy_coords),
        outer_airfoil=shapely.geometry.LineString(
            wing.sections[idx + 1].airfoil.xy_coords
        ),
        deltaZ=deltaZ,
        alpha_inner=wing.sections[idx].angle,
        alpha_outer=wing.sections[idx + 1].angle,
        inner_le_x=wing.sections[idx].x_le,
        inner_le_y=wing.sections[idx].z_le,
        outer_le_x=wing.sections[idx + 1].x_le,
        outer_le_y=wing.sections[idx + 1].z_le,
        inner_chord=wing.sections[idx].chord,
        outer_chord=wing.sections[idx + 1].chord,
    )

    print(section)
    # plot_2d(wing)
    machine_def = tomllib.loads(pathlib.Path("machine.toml").read_text())
    deltaZ
    # override deltaZ because we will be cutting that
    machine_def["stock"]["volume"]["deltaZ"] = (
        deltaZ * machine_def["stock"]["offset"]["side"]
    )
    # machine_def["stock"]["volume"]["deltaY"] *= machine_def["stock"]["offset"]["side"]
    cut_section(section, machine_def)
