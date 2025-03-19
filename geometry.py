import re
from dataclasses import dataclass, field
from typing import List, Optional
import rich.traceback
from collections import deque
from rich import print
import numpy as np

DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def get_rot_mat(alpha):
    """Create a rotation matrix for rotation around the y-axis."""
    angle_rad = np.radians(alpha)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])


def get_rot_mat2d(alpha):
    """Create a rotation matrix for rotation around the y-axis."""
    angle_rad = np.radians(alpha)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos_a, sin_a], [-sin_a, cos_a]])


@dataclass
class Airfoil:
    name: str
    xy_coords: np.ndarray
    xz_coords: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    top_coords: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    btm_coords: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self):
        self.xz_coords = np.insert(self.xy_coords, 1, 0, axis=1)
        leading_edge_idx = np.argmin(self.xz_coords[:, 0])
        self.top_coords = self.xz_coords[: leading_edge_idx + 1][::-1]
        self.btm_coords = self.xz_coords[leading_edge_idx:]

    @staticmethod
    def get_transformed_coords(coords, chord, alpha, leading_edge_vec):
        return chord * (get_rot_mat(alpha) @ coords.T).T + leading_edge_vec

    @staticmethod
    def get_plot_2d_args(coords):
        return (
            coords[:, 0],
            coords[:, 2],
        )

    @staticmethod
    def get_plot_3d_args(coords):
        return (
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        )

    @classmethod
    def from_dat(cls, filename: str):
        coords = np.genfromtxt(filename, skip_header=True)

        with open(filename, "r") as f:
            name = f.readlines()[0].strip()

        return cls(name, coords)


@dataclass
class Section:
    # leading_edge_vec: np.ndarray = field(default_factory=lambda: np.array([],dtype=float))
    x_le: float
    y_le: float
    z_le: float
    chord: float
    angle: float
    nspan: Optional[int] = None
    sspace: Optional[float] = None
    airfoil_file: Optional[str] = None
    airfoil: Optional[Airfoil] = None

    def get_transform(self):
        return np.array([[self.x_le, self.y_le, self.z_le]])

    def get_rot_mat(self):
        """Create a rotation matrix for rotation around the y-axis."""
        return get_rot_mat(self.angle)

    def get_transformed_xy_coords(self):
        return self.chord * (
            get_rot_mat2d(self.angle) @ self.airfoil.xy_coords.T
        ).T + np.array([[self.x_le, self.z_le]])


@dataclass
class Surface:
    name: str
    nchord: int
    cspace: float
    nspan: Optional[int] = None
    sspace: Optional[float] = None
    y_duplicate: Optional[float] = None
    angle: Optional[float] = None
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    translate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sections: List[Section] = field(default_factory=list)


def parse_header(lines: deque[str]):
    name = lines.popleft()
    mach = lines.popleft().split()[0]
    iYsym, iZsym, Zsym = lines.popleft().split()[:3]
    Sref, Cref, Bref = lines.popleft().split()[:3]
    Xref, Yref, Zref = lines.popleft().split()[:3]
    CDoref = lines.popleft().split()[0]

    print_debug("name =", name)
    print_debug("mach =", mach)
    print_debug("iYsym, iZsym ,Zsym =", iYsym, iZsym, Zsym)
    print_debug("Sref, Cref, Bref =", Sref, Cref, Bref)
    print_debug("Xref, Yref, Zref =", Xref, Yref, Zref)
    print_debug("CDoref =", CDoref)


def parse_surface(lines: deque[str]) -> Surface:
    print_debug("parse_surface")
    ident = lines.popleft()
    surface_name = lines.popleft()
    Nspan, Sspace = None, None
    Nchord, Cspace, *rest = [float(x) for x in lines.popleft().split()]
    if len(rest) > 1:
        Nspan = rest[0]
    if len(rest) == 2:
        Sspace = rest[1]

    print_debug(
        f"SURFACE - {surface_name!r} - Nchord, Cspace, Nspan, Sspace =",
        Nchord,
        Cspace,
        Nspan,
        Sspace,
    )

    sections = []
    ydup = 1.0
    dAinc = 0.0
    Xscale, Yscale, Zscale = 1.0, 1.0, 1.0
    dX, dY, dZ = 0.0, 0.0, 0.0

    while lines:
        # this is no longer part of the current surface definition
        if lines[0].lower().startswith("surf") or lines[0].lower().startswith("body"):
            break

        if lines[0].lower().startswith("comp") or lines[0].lower().startswith("inde"):
            ident = lines.popleft()
            Lcomp = float(lines.popleft())
            continue

        if lines[0].lower().startswith("ydup"):
            ident = lines.popleft()
            ydup = float(lines.popleft())
            print_debug("ydup = ", ydup)
            continue

        if lines[0].lower().startswith("scal"):
            ident = lines.popleft()
            Xscale, Yscale, Zscale = [float(x) for x in lines.popleft().split()]
            print_debug("scal = ", Xscale, Yscale, Zscale)
            continue

        if lines[0].lower().startswith("tran"):
            ident = lines.popleft()
            dX, dY, dZ = [float(x) for x in lines.popleft().split()]
            print_debug("tran = ", dX, dY, dZ)
            continue

        if lines[0].lower().startswith("angl") or lines[0].lower().startswith("ainc"):
            ident = lines.popleft()
            dAinc = float(lines.popleft())
            print_debug("dAinc = ", dAinc)
            continue

        if lines[0].lower().startswith("nowa"):
            ident = lines.popleft()
            print_debug("nowa")
            continue

        if lines[0].lower().startswith("noab"):
            ident = lines.popleft()
            print_debug("noab")
            continue

        if lines[0].lower().startswith("nolo"):
            ident = lines.popleft()
            print_debug("nolo")
            continue

        if lines[0].lower().startswith("cdcl"):
            ident = lines.popleft()
            CL1, CD1, CL2, CD2, CL3, CD3 = [float(x) for x in lines.popleft().split()]
            print_debug("cdcl = ", CL1, CD1, CL2, CD2, CL3, CD3)
            continue

        if lines[0].lower().startswith("sect"):
            sections.append(
                parse_section(
                    lines, surface_name=surface_name, Nspan=Nspan, Sspace=Sspace
                )
            )
            continue
    return Surface(
        surface_name,
        Nchord,
        Cspace,
        Nspan,
        Sspace,
        ydup,
        dAinc,
        [Xscale, Yscale, Zscale],
        [dX, dY, dZ],
        sections,
    )


def parse_section(lines: deque[str], surface_name, Nspan=None, Sspace=None) -> Section:
    ident = lines.popleft()
    Xle, Yle, Zle, Chord, Ainc, *rest = [float(x) for x in lines.popleft().split()]
    if len(rest) > 1:
        Nspan = rest[0]
    if len(rest) == 2:
        Sspace = rest[1]

    if Nspan is None:
        raise ValueError(
            f"Nspan is not defined for the SURFACE, therefor it must be defined for the SECTION. In SURFACE {surface_name!r}"
        )

    if Sspace is None:
        raise ValueError(
            f"Sspace is not defined for the SURFACE, therefor it must be defined for the SECTION. In SURFACE {surface_name!r}"
        )

    filename = None
    airfoil = None

    print_debug(
        "SECTION - Xle, Yle, Zle, Chord, Ainc, Nspan, Sspace",
        Xle,
        Yle,
        Zle,
        Chord,
        Ainc,
        Nspan,
        Sspace,
    )
    while lines:
        if lines[0].lower().startswith("naca"):
            ident = lines.popleft()
            raise NotImplementedError(
                "Parsing for the NACA keyword is not yet implemented"
            )

        if lines[0].lower().startswith("airf"):
            raise NotImplementedError(
                "Parsing for the AIRFOIL keyword is not yet implemented"
            )

        if lines[0].lower().startswith("afil"):
            ident, *rest = lines.popleft().split()
            if rest:
                raise NotImplementedError(
                    "Parsing for the X1 and X2 param in AFILE section is not yet implemented"
                )
            filename = lines.popleft()
            print_debug("filename = ", filename)
            airfoil = Airfoil.from_dat(filename)
            continue

        if lines[0].lower().startswith("cont"):
            ident = lines.popleft()
            name, *params = lines.popleft().split()
            gain, Xhinge, XYZhvecX, XYZhvecY, XYZhvecZ, *rest = [
                float(x) for x in params
            ]
            XYZhvec = XYZhvecX, XYZhvecY, XYZhvecZ
            SgnDup = 1
            if rest:
                SgnDup = rest[0]
            print_debug("contol = ", name, gain, Xhinge, XYZhvec, SgnDup)
            continue

        break

    # all subsections accounted for, the current token must be part of a diffrent section. Break out and pass evaluation back to the caller
    return Section(Xle, Yle, Zle, Chord, Ainc, Nspan, Sspace, filename, airfoil)


def get_wing_surface_from_avl_file(filename: str) -> Surface:
    with open(filename, "r") as file:
        lines = file.readlines()

    surface = None
    capturing = False

    # discard text to the right of the comment character
    lines = [line.split("!")[0].strip() for line in lines]
    # confusingly lines that start with '#' are also comments
    lines = [line for line in lines if not line.startswith("#")]
    # drop empty lines
    lines = [line for line in lines if line]

    lines = deque(lines)

    parse_header(lines)

    while lines:
        line = lines[0]

        if line.lower().startswith("surf"):
            _surface = parse_surface(lines)
            if _surface.name == "Wing":
                surface = _surface
            continue

        # no matches consume the line
        lines.popleft()

    if surface is None:
        raise ValueError("Could not find surface named 'Wing'")

    return surface


if __name__ == "__main__":
    rich.traceback.install()
    # Example usage
    wing_surface = get_wing_surface_from_avl_file("supergee.avl")
    print(wing_surface)
