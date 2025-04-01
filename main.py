from dataclasses import dataclass
from typing import Optional
from xml.etree.ElementTree import TreeBuilder

import shapely.ops
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

inch = 25.4

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

        airfoil = shapely.ops.transform(lambda x, y: (x, y, 0.0), airfoil)
        return airfoil
def compute_gcode(
    left_top_cut: shapely.geometry.LineString,
    right_top_cut: shapely.geometry.LineString,
    left_btm_cut: shapely.geometry.LineString,
    right_btm_cut: shapely.geometry.LineString,
    trailing_edge: Optional[tuple[float]] = None,
):
    
    if not trailing_edge:
        trailing_edge = (
            max([x for (x, _, _) in left_top_cut.coords]),
            max([x for (x, _, _) in right_top_cut.coords])
        )
    gcode = """\
(Program Start)
G17 G21 G90 G40 G49 G64
(Initial Height)
"""

    RAPID_HEIGHT = 3
    FEED = 200

    y, z = 0.0, 0.0
    x, u = 0, 0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = RAPID_HEIGHT, RAPID_HEIGHT
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"

    x, u = trailing_edge
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = 0.0, 0.0
    gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    x, u = 10, 10
    gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = RAPID_HEIGHT, RAPID_HEIGHT
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    x, u = 0, 0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = 0.0, 0.0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"



    assert len(left_top_cut.coords) == len(right_top_cut.coords)

    (_, y, _), (_, z, _) = left_top_cut.coords[0], right_top_cut.coords[0]

    # Add rapid move up to Y and Z
    gcode += f"G0 Y{y * inch:06.2f} Z{z * inch:06.2f}\n"
    for (x, y, _), (u, z, _) in zip(left_top_cut.coords, right_top_cut.coords):
        gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"


    y, z = RAPID_HEIGHT, RAPID_HEIGHT
    gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    x, u = 0, 0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = 0.0, 0.0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"



    assert len(left_btm_cut.coords) == len(right_btm_cut.coords)
    (_, y, _), (_, z, _) = left_btm_cut.coords[0], right_btm_cut.coords[0]

    # Add rapid move up to Y and Z
    gcode += f"G0 Y{y * inch:06.2f} Z{z * inch:06.2f}\n"
    for (x, y, _), (u, z, _) in zip(left_btm_cut.coords, right_btm_cut.coords):
        gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = RAPID_HEIGHT, RAPID_HEIGHT
    gcode += f"G1 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    x, u = 0, 0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    y, z = 0.0, 0.0
    gcode += f"G0 X{x * inch:06.2f} Y{y * inch:06.2f} A{u * inch:06.2f} Z{z * inch:06.2f} F{FEED}\n"
    gcode += "M2\n"

    return gcode

def cut_section(section: WingSection, machine_def):

    plotter = pv.Plotter()
    setup_scene(plotter, machine_def)

    INSET_AIRFOIL = 0.25

    trailing_edge = (
        section.inner_le_x + section.inner_chord + INSET_AIRFOIL + machine_def["stock"]["offset"]["absoluteX"],
        section.outer_le_x + section.outer_chord + INSET_AIRFOIL + machine_def["stock"]["offset"]["absoluteX"]
        )
    
    # wing_csys to stock cysy
    inner_airfoil = shapely.affinity.translate(
            section._transformed_inner_airfoil,
            xoff=INSET_AIRFOIL, 
            yoff=machine_def["cut"]["leading_edge_offsetY"], 
            zoff=0
        )
    outer_airfoil = shapely.affinity.translate(
            section._transformed_outer_airfoil,
            xoff=INSET_AIRFOIL, 
            yoff=machine_def["cut"]["leading_edge_offsetY"], 
            zoff= -section.deltaZ
        )
    
    # stock csys to machine csys
    inner_airfoil = shapely.affinity.translate(
            inner_airfoil,
            xoff=machine_def["stock"]["offset"]["absoluteX"], 
            yoff=machine_def["stock"]["offset"]["absoluteY"], 
            zoff=machine_def["stock"]["offset"]["absoluteZ"]
        )
    
    outer_airfoil = shapely.affinity.translate(
        outer_airfoil,
        xoff=machine_def["stock"]["offset"]["absoluteX"], 
        yoff=machine_def["stock"]["offset"]["absoluteY"], 
        zoff=machine_def["stock"]["offset"]["absoluteZ"]
    )
    

    points = shapely.get_coordinates(inner_airfoil, include_z=True)
    inner_top_cut, inner_btm_cut = cutpath.compute_cutpaths(
        points,
        kerf=machine_def["cut"]["kerf"],
        start_point=machine_def["stock"]["offset"]["absoluteX"] - 0,
    )
    l = pv.lines_from_points(points)
    plotter.add_mesh(l, color="#fc33ff")

    points = shapely.get_coordinates(outer_airfoil, include_z=True)
    outer_top_cut, outer_btm_cut = cutpath.compute_cutpaths(
        points,
        kerf=machine_def["cut"]["kerf"],
        start_point=machine_def["stock"]["offset"]["absoluteX"] - 0,
    )
    l = pv.lines_from_points(
        points,
    )
    plotter.add_mesh(l, color="b")

    inner_top_code, outer_top_code = cutpath.project_cut_paths_to_planes(
        inner_top_cut,
        outer_top_cut,
        machine_def["stock"]["offset"]["absoluteZ"],
        machine_def["stock"]["offset"]["absoluteZ"] - section.deltaZ,
        0,
        -machine_def["work"]["volume"]["deltaZ"],
    )
    left_btm_code, right_btm_code = cutpath.project_cut_paths_to_planes(
        inner_btm_cut,
        outer_btm_cut,
        machine_def["stock"]["offset"]["absoluteZ"],
        machine_def["stock"]["offset"]["absoluteZ"] - section.deltaZ,
        0,
        -machine_def["work"]["volume"]["deltaZ"],
    )

    for linestring in [inner_top_code, outer_top_code, left_btm_code, right_btm_code]:
        l = pv.lines_from_points(
            shapely.get_coordinates(linestring, include_z=True),
        )
        plotter.add_mesh(l, color="g")
    plotter.show()

    gcode = compute_gcode(
        inner_top_code,
        outer_top_code,
        left_btm_code,
        right_btm_code,
        trailing_edge=trailing_edge
    )
    import pathlib

    pathlib.Path("out.nc").write_text(gcode)

if __name__ == "__main__":
    rich.traceback.install()
    machine_def = tomllib.loads(pathlib.Path("machine.toml").read_text())

    # Example usage
    wing = geometry.get_wing_surface_from_avl_file("synergyII\synergyIImod.avl")

    idx = machine_def["cut"]["section"]
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
    
    # override deltaZ because we will be cutting that
    machine_def["stock"]["volume"]["deltaZ"] = deltaZ
    if machine_def["stock"]["offset"]["absoluteZ"] > 0:
        machine_def["stock"]["offset"]["absoluteZ"] = -machine_def["work"]["volume"]["deltaZ"] + machine_def["stock"]["offset"]["absoluteZ"] + machine_def["stock"]["volume"]["deltaZ"]
        
    cut_section(section, machine_def)
