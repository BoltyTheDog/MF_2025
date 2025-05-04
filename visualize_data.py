#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse


def load_scalar_data(filename):
    """Load scalar field data produced by the Go simulation."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                x, y, val = map(float, line.split())
                data.append((x, y, val))

    # Extract unique x and y coordinates to determine grid size
    x_coords = sorted(set(point[0] for point in data))
    y_coords = sorted(set(point[1] for point in data))
    nx, ny = len(x_coords), len(y_coords)

    # Create 2D arrays
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    Z = np.zeros((ny, nx))

    # Map each data point to its position in the grid
    x_indices = {x: i for i, x in enumerate(x_coords)}
    y_indices = {y: j for j, y in enumerate(y_coords)}

    for x, y, val in data:
        i, j = x_indices[x], y_indices[y]
        X[j, i] = x
        Y[j, i] = y
        Z[j, i] = val

    return X, Y, Z


def load_vector_data(filename):
    """Load vector field data produced by the Go simulation."""
    x, y, u, v = [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                values = list(map(float, line.split()))
                x.append(values[0])
                y.append(values[1])
                u.append(values[2])
                v.append(values[3])

    return np.array(x), np.array(y), np.array(u), np.array(v)


def visualize_scalar_field(filename, title, colormap='coolwarm'):
    """Visualize a scalar field from the data file."""
    print(f"Loading data from {filename}...")
    X, Y, Z = load_scalar_data(filename)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, 50, cmap=colormap)
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    output_file = filename.replace('.data', '_viz.png')
    plt.savefig(output_file, dpi=150)
    print(f"Visualization saved to {output_file}")
    plt.close()


def visualize_vector_field(filename, title):
    """Visualize a vector field from the data file."""
    print(f"Loading data from {filename}...")
    x, y, u, v = load_vector_data(filename)

    # Calculate velocity magnitude
    magnitude = np.sqrt(u ** 2 + v ** 2)

    plt.figure(figsize=(10, 8))

    # Plot velocity vectors
    plt.quiver(x, y, u, v, magnitude, cmap='viridis', scale=25)
    plt.colorbar(label='Velocity Magnitude')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    output_file = filename.replace('.data', '_viz.png')
    plt.savefig(output_file, dpi=150)
    print(f"Visualization saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize fluid flow simulation data')
    parser.add_argument('--stream', default='stream_function.data',
                        help='Stream function data file')
    parser.add_argument('--velocity', default='velocity_magnitude.data',
                        help='Velocity magnitude data file')
    parser.add_argument('--pressure', default='pressure.data',
                        help='Pressure field data file')
    parser.add_argument('--vectors', default='velocity_field.data',
                        help='Velocity field vectors data file')

    args = parser.parse_args()

    visualize_scalar_field(args.stream, 'Stream Function')
    visualize_scalar_field(args.velocity, 'Velocity Magnitude', 'viridis')
    visualize_scalar_field(args.pressure, 'Pressure Field', 'rainbow')
    visualize_vector_field(args.vectors, 'Velocity Field')


if __name__ == '__main__':
    main()