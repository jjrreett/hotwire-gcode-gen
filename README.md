# Hot Wire GCODE Generator

For rcKeith's GRBL Hot Wire setup

Define geometry using the Athena Vortex Latices file format. See https://web.mit.edu/drela/Public/web/avl/


## Coordinate Systems
### AVL Coordinate System
The geometry is described in the following Cartesian system:
- X - downstream
- Y - out the right wing
- Z - up

### Machine Coordinate System (CSYS)
- X - downstream
- Y - up
- Z - out the left wing

### GCODE/Motor Axis
- X - positive X axis motion correspond to the **left** side of the wire moving **downstream**
- Y - positive Y axis motion correspond to the **left** side of the wire moving **up**
- Z - positive Z axis motion correspond to the **right** side of the wire moving **downstream**
- A - positive A axis motion correspond to the **right** side of the wire moving **up**

If you set the origin of the Machine CSYS to be on the Motor Axis XY plane, the the work volume is in the -Z direction
