import rich.traceback
from rich import print
import geometry
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    rich.traceback.install()
    # Example usage
    wing = geometry.get_wing_surface_from_avl_file("supergee.avl")
    plot_2d(wing)
