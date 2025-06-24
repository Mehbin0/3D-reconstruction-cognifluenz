import matplotlib.pyplot as plt
import pyvista as pv

# Function to read and parse points3D.txt
def read_points(file_path):
    points = []
    colors = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines
            parts = line.split()
            if len(parts) < 7:
                continue  # Skip lines that don't have enough data
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            points.append((x, y, z))
            colors.append((r / 255, g / 255, b / 255))  # Normalize RGB values
    return points, colors

# Function to visualize 3D points
def visualize_points(points, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c=colors, marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def export_points_to_file(points, colors, output_file):
    """Export points and colors to a file."""
    with open(output_file, 'w') as file:
        for (x, y, z), (r, g, b) in zip(points, colors):
            file.write(f"{x} {y} {z} {int(r * 255)} {int(g * 255)} {int(b * 255)}\n")

def export_points_to_ply(points, colors, output_file):
    """Export points and colors to a PLY file."""
    with open(output_file, 'w') as file:
        # Write PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        # Write point data
        for (x, y, z), (r, g, b) in zip(points, colors):
            file.write(f"{x} {y} {z} {int(r * 255)} {int(g * 255)} {int(b * 255)}\n")

def visualize_surface_with_pyvista(points, colors):
    """Visualize 3D points as a colored point cloud using PyVista."""
    # Create a PyVista point cloud
    point_cloud = pv.PolyData(points)
    point_cloud['colors'] = [tuple(int(c * 255) for c in color) for color in colors]

    # Plot the point cloud
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=5)
    plotter.show(title="3D Surface Visualization with PyVista")

def create_surface_mesh(points):
    """Create a surface mesh from the point cloud using Delaunay triangulation."""
    import numpy as np
    from scipy.spatial import Delaunay

    # Convert points to a NumPy array
    points_array = np.array(points)

    # Perform Delaunay triangulation
    tri = Delaunay(points_array[:, :2])  # Use X and Y coordinates for triangulation

    # Create a PyVista mesh
    mesh = pv.PolyData(points_array)
    mesh.faces = np.hstack((np.full((tri.simplices.shape[0], 1), 3), tri.simplices)).flatten()

    return mesh

def visualize_surface_mesh_with_pyvista(points, colors):
    """Visualize the surface mesh with PyVista."""
    # Create the surface mesh
    mesh = create_surface_mesh(points)

    # Add colors to the mesh
    mesh['colors'] = [tuple(int(c * 255) for c in color) for color in colors]

    # Plot the surface mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colors', rgb=True)
    plotter.show(title="3D Surface Mesh Visualization with PyVista")

def visualize_depth_shading(points):
    """Visualize 3D points with depth-based shading."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_heatmap(points):
    """Visualize 3D points with a heatmap based on depth."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c=zs, cmap='hot', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_voxelization(points):
    """Visualize the voxelized representation of the surface mesh."""
    import numpy as np
    import pyvista as pv

    # Create a surface mesh using Delaunay triangulation
    mesh = create_surface_mesh(points)

    # Generate a voxelized representation manually
    bounds = mesh.bounds
    grid = pv.UniformGrid()
    grid.dimensions = [50, 50, 50]  # Define the resolution of the voxel grid
    grid.origin = [bounds[0], bounds[2], bounds[4]]  # Set the origin
    grid.spacing = [(bounds[1] - bounds[0]) / 50, (bounds[3] - bounds[2]) / 50, (bounds[5] - bounds[4]) / 50]  # Set the spacing

    # Plot the voxel grid
    plotter = pv.Plotter()
    plotter.add_mesh(grid, style='wireframe')
    plotter.show(title="Voxelized Representation")

def print_visualization_type(visualization_name):
    """Print the type of visualization being loaded."""
    print(f"Loading visualization: {visualization_name}")

def main():
    file_path = "points3D.txt"  # Path to the points3D.txt file
    output_file = "exported_points.txt"  # Path to the output file
    ply_output_file = "exported_points.ply"  # Path to the PLY output file
    points, colors = read_points(file_path)

    print_visualization_type("3D Points")
    visualize_points(points, colors)

    print_visualization_type("Surface Point Cloud")
    visualize_surface_with_pyvista(points, colors)

    print_visualization_type("Surface Mesh")
    visualize_surface_mesh_with_pyvista(points, colors)

    print_visualization_type("Depth Shading")
    visualize_depth_shading(points)

    print_visualization_type("Heatmap")
    visualize_heatmap(points)

    export_points_to_file(points, colors, output_file)
    export_points_to_ply(points, colors, ply_output_file)
    print(f"Points exported to {output_file} and {ply_output_file}")

if __name__ == "__main__":
    main()