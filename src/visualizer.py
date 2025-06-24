import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from config import DATASET_NAME

def load_point_cloud(filename):
    """Load 3D points from file"""
    
    file_path = Path(f"../data/{DATASET_NAME}/{filename}")
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                coords = line.split()
                if len(coords) >= 3:
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    points.append([x, y, z])
    
    points = np.array(points)
    print(f"Loaded {len(points)} 3D points")
    return points

def visualize_point_cloud_basic(points_3d):
    """Basic 3D scatter plot"""
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by depth (Z coordinate)
    scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                        c=points_3d[:, 2], cmap='viridis', s=50, alpha=0.8)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (depth)')
    ax.set_title(f'Statue 3D Reconstruction\n{len(points_3d)} Points')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Depth (Z)')
    
    # Equal aspect ratio
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                         points_3d[:, 1].max() - points_3d[:, 1].min(),
                         points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def visualize_multiple_views(points_3d):
    """Show multiple viewing angles of the point cloud"""
    
    fig = plt.figure(figsize=(15, 12))
    
    # Multiple viewing angles
    angles = [
        (30, 45, "Front-Right View"),
        (30, 135, "Back-Right View"), 
        (30, 225, "Back-Left View"),
        (30, 315, "Front-Left View"),
        (0, 0, "Front View"),
        (90, 0, "Top View")
    ]
    
    for i, (elev, azim, title) in enumerate(angles):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        scatter = ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           c=points_3d[:, 2], cmap='plasma', s=30, alpha=0.7)
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    plt.suptitle(f'Statue 3D Point Cloud - Multiple Views\n{len(points_3d)} Points', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_point_distribution(points_3d):
    """Analyze the distribution of 3D points"""
    
    print("\n📊 === Point Cloud Analysis ===")
    
    # Basic statistics
    print(f"Total points: {len(points_3d)}")
    print(f"Coordinate ranges:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        min_val, max_val = points_3d[:, i].min(), points_3d[:, i].max()
        mean_val, std_val = points_3d[:, i].mean(), points_3d[:, i].std()
        print(f"  {axis}: {min_val:.2f} to {max_val:.2f} (mean: {mean_val:.2f}, std: {std_val:.2f})")
    
    # Point density analysis
    center = np.mean(points_3d, axis=0)
    distances = np.linalg.norm(points_3d - center, axis=1)
    
    print(f"\nPoint cloud center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"Average distance from center: {distances.mean():.2f}")
    print(f"Point cloud radius: {distances.max():.2f}")
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coordinate histograms
    for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], ['red', 'green', 'blue'])):
        row, col = i // 2, i % 2
        if i < 3:
            axes[row, col].hist(points_3d[:, i], bins=20, alpha=0.7, color=color)
            axes[row, col].set_title(f'{axis} Coordinate Distribution')
            axes[row, col].set_xlabel(f'{axis} value')
            axes[row, col].set_ylabel('Count')
    
    # Distance from center
    axes[1, 1].hist(distances, bins=20, alpha=0.7, color='purple')
    axes[1, 1].set_title('Distance from Center')
    axes[1, 1].set_xlabel('Distance')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def visualize_with_connections(points_3d):
    """Try to show potential structure by connecting nearby points"""
    
    from scipy.spatial.distance import pdist, squareform
    
    print("\n🔗 Attempting to show structure...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Just points (what we had)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c=points_3d[:, 2], cmap='viridis', s=50)
    ax1.set_title('Sparse Points Only')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Plot 2: Connect nearby points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c=points_3d[:, 2], cmap='viridis', s=50)
    
    # Connect points that are close to each other
    distances = squareform(pdist(points_3d))
    threshold = np.percentile(distances[distances > 0], 5)  # Connect closest 5%
    
    for i in range(len(points_3d)):
        for j in range(i+1, len(points_3d)):
            if distances[i, j] < threshold:
                ax2.plot([points_3d[i,0], points_3d[j,0]], 
                        [points_3d[i,1], points_3d[j,1]], 
                        [points_3d[i,2], points_3d[j,2]], 'gray', alpha=0.3, linewidth=0.5)
    
    ax2.set_title('With Connections')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # Plot 3: Show point density regions
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Color by local density
    densities = []
    for i, point in enumerate(points_3d):
        nearby_count = np.sum(distances[i] < threshold * 2)
        densities.append(nearby_count)
    
    scatter = ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                         c=densities, cmap='hot', s=60)
    ax3.set_title('Point Density\n(Red = Dense Areas)')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    
    plt.colorbar(scatter, ax=ax3, label='Local Density')
    plt.tight_layout()
    plt.show()
    
    print(f"Connection threshold: {threshold:.2f}")
    print("Dense (red) areas likely represent statue features")

def compare_with_simple_shapes(points_3d):
    """Compare point cloud dimensions with simple shapes"""
    
    print("\n📏 === Shape Analysis ===")
    
    # Calculate bounding box
    min_coords = np.min(points_3d, axis=0)
    max_coords = np.max(points_3d, axis=0)
    dimensions = max_coords - min_coords
    
    print(f"Bounding box dimensions:")
    print(f"  Width (X):  {dimensions[0]:.2f}")
    print(f"  Height (Y): {dimensions[1]:.2f}")
    print(f"  Depth (Z):  {dimensions[2]:.2f}")
    
    # Guess the type of object
    ratios = dimensions / np.min(dimensions)
    print(f"\nDimension ratios: {ratios}")
    
    if ratios[1] > 2:  # Height much larger
        print("🗿 Shape suggests: Tall statue (good!)")
    elif max(ratios) < 1.5:  # All dimensions similar
        print("🟫 Shape suggests: Cubic/spherical object")
    else:
        print("📐 Shape suggests: General 3D object")
    
    # Show 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY projection (top view)
    axes[0].scatter(points_3d[:, 0], points_3d[:, 1], c=points_3d[:, 2], cmap='viridis', s=30)
    axes[0].set_title('Top View (XY projection)')
    axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')
    
    # XZ projection (side view)
    axes[1].scatter(points_3d[:, 0], points_3d[:, 2], c=points_3d[:, 1], cmap='plasma', s=30)
    axes[1].set_title('Side View (XZ projection)')
    axes[1].set_xlabel('X'); axes[1].set_ylabel('Z')
    axes[1].set_aspect('equal')
    
    # YZ projection (front view)
    axes[2].scatter(points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 0], cmap='coolwarm', s=30)
    axes[2].set_title('Front View (YZ projection)')
    axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎨 === Enhanced 3D Point Cloud Visualizer ===")
    
    # Load the point cloud
    points_3d = load_point_cloud("statue_advanced_3d.txt")
    
    if points_3d is not None:
        print("\n1. Basic 3D visualization")
        visualize_point_cloud_basic(points_3d)
        
        print("\n2. Enhanced structure analysis")
        visualize_with_connections(points_3d)
        
        print("\n3. Shape analysis and 2D projections")
        compare_with_simple_shapes(points_3d)
        
        print("\n4. Statistical analysis")
        analyze_point_distribution(points_3d)
        
        print("\n✨ Analysis complete!")
        
        print(f"\n💡 Remember: 210 points is very sparse!")
        print(f"   Dense reconstruction will have 1000s of points for recognizable shape")
    else:
        print("❌ Could not load point cloud file")