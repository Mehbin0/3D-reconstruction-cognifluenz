# Refactored visualizer script
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATASET_NAME

# Visualize point cloud

def visualize_point_cloud(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    plt.show()

# Load and visualize 3D point cloud

def load_point_cloud(file_path):
    points_3d = np.loadtxt(file_path, delimiter=" ")
    print(f"Loaded {len(points_3d)} points from {file_path}")
    return points_3d

# Load and visualize 3D point cloud from PLY file

def load_point_cloud_from_ply(file_path):
    points_3d = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("end_header"):
                break
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3:
                points_3d.append([float(parts[0]), float(parts[1]), float(parts[2])])
    points_3d = np.array(points_3d)
    print(f"Loaded {len(points_3d)} points from PLY file: {file_path}")
    return points_3d

if __name__ == "__main__":
    file_format = "txt"  # Change to "ply" to load PLY file
    if file_format == "txt":
        point_cloud_path = Path(f"../data/{DATASET_NAME}/points3D.txt")
        points_3d = load_point_cloud(point_cloud_path)
    elif file_format == "ply":
        ply_file_path = Path(f"../data/{DATASET_NAME}/points3D.ply")
        points_3d = load_point_cloud_from_ply(ply_file_path)
    visualize_point_cloud(points_3d)