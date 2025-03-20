import pyvista as pv
import numpy as np

# Create a PyVista plotter
plotter = pv.Plotter()

# Define Machine Volume (cutting area)
machine_bounds = pv.Box(bounds=(-20, 20, -20, 20, -10, 10))
plotter.add_mesh(machine_bounds, color="gray", opacity=0.3, style="wireframe")

# Define Stock Volume (material to be cut)
stock = pv.Box(bounds=(-5, 5, -5, 5, -5, 5))
plotter.add_mesh(stock, color="brown", opacity=0.5)

# Sample G-code toolpath (wire endpoints at each step)
toolpath = [
    [(0, -5, 0), (10, -5, 0)],
    [(0, 0, 5), (10, 0, 5)],
    [(0, 5, 0), (10, 5, 0)],
]

# Draw toolpath
for p1, p2 in toolpath:
    line = pv.Line(p1, p2)
    plotter.add_mesh(line, color="blue", line_width=3)

# Show the visualization
plotter.show()
