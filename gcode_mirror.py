import re
import argparse
import os

def swap_coordinates(gcode_line):
    # Extract all coordinates using regex
    matches = re.findall(r'([XYZA])([-+]?[0-9]*\.?[0-9]+)', gcode_line)
    coord_map = dict(matches)

    # Apply the swap: X <-> A and Y <-> Z
    swapped = {
        'X': coord_map.get('A'),
        'Y': coord_map.get('Z'),
        'Z': coord_map.get('Y'),
        'A': coord_map.get('X'),
    }

    # Rebuild the line
    def repl(match):
        axis = match.group(1)
        if swapped.get(axis) is not None:
            return f'{axis}{swapped[axis]}'
        return match.group(0)

    return re.sub(r'([XYZA])([-+]?[0-9]*\.?[0-9]+)', repl, gcode_line)

def process_gcode_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.strip().startswith(('G', 'M')):  # only process motion/control lines
                new_line = swap_coordinates(line)
                outfile.write(new_line)
            else:
                outfile.write(line)

def main():
    parser = argparse.ArgumentParser(description='Swap X<->A and Y<->Z in G-code.')
    parser.add_argument('input', help='Path to input G-code file')
    parser.add_argument('-o', '--output', help='Path to output file (default: input_swapped.gcode)')

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output or f"{os.path.splitext(input_path)[0]}_mirror.gcode"

    process_gcode_file(input_path, output_path)
    print(f"✅ Swapped G-code written to: {output_path}")

if __name__ == '__main__':
    main()
