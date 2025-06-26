import matplotlib.pyplot as plt
import pyvista as pv

def visualize_point_cloud(points_3d):
    """Visualize 3D point cloud."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def visualize_surface_cloud(points_3d):
    """Visualize 3D surface cloud."""
    cloud = pv.PolyData(points_3d)
    surface = cloud.delaunay_2d()
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color="lightblue")
    plotter.show()

def visualize_menu(points_3d):
    """Menu for visualization options."""
    print("\nWould you like to visualize the 3D point cloud?")
    print("1. Point Cloud")
    print("2. Surface Cloud")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")
    if choice == "1":
        visualize_point_cloud(points_3d)
    elif choice == "2":
        visualize_surface_cloud(points_3d)
    else:
        print("Exiting without visualization.")
