import cv2
import numpy as np
import sys
from pathlib import Path
from config import DATASET_NAME

# Fix import paths
sys.path.append(str(Path(__file__).parent))

try:
    from pose_estimator import load_camera_matrix, estimate_camera_pose
    from feature_matcher import match_features_between_images
    from triangulation import triangulate_points
except ImportError as e:
    print(f"⚠️  Import Error: {e}")
    print("Make sure pose_estimator.py, feature_matcher.py, and triangulation.py are in the same directory")
    sys.exit(1)

def validate_data_setup():
    """Check if all required data is available"""
    
    data_path = Path(f"../data/{DATASET_NAME}") 
    issues = []
    
    if not data_path.exists():
        issues.append("Data directory not found")
        return issues
    
    images_path = data_path / "images" / "dslr_images_undistorted"
    if not images_path.exists():
        issues.append("Images directory not found")
    else:
        image_files = list(images_path.glob("*.JPG"))
        if len(image_files) < 2:
            issues.append(f"Need at least 2 images, found {len(image_files)}")
    
    calibration_path = data_path / "calibration"
    if not calibration_path.exists():
        issues.append("Calibration directory not found")
    
    return issues

def compute_depth_safely(disparity, focal_length, baseline, min_disp=1, max_depth=100):
    """Safely compute depth with realistic bounds"""
    valid_pixels = disparity > min_disp
    depth_map = np.zeros_like(disparity, dtype=np.float32)
    
    if np.sum(valid_pixels) > 0:
        depths = (focal_length * baseline) / (disparity[valid_pixels] + 1e-6)
        realistic_depths = np.clip(depths, 0, max_depth)
        realistic_depths[depths >= max_depth] = 0
        depth_map[valid_pixels] = realistic_depths
    
    return depth_map

def compute_improved_depth_map(image1_path, image2_path, camera_matrix, R, t):
    """Compute depth map using improved stereo matching"""
    
    print(f"Computing depth map: {image1_path.name} + {image2_path.name}")
    
    # Load images
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("❌ Error loading images")
        return None, None, None
    
    # Resize for processing
    scale = 0.5
    h, w = img1.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    img1_small = cv2.resize(img1, (new_w, new_h))
    img2_small = cv2.resize(img2, (new_w, new_h))
    
    print(f"Processing at resolution: {new_w}x{new_h}")
    
    # Use StereoSGBM (best results)
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo_sgbm.compute(img1_small, img2_small)
    
    # Convert to depth
    focal_length = camera_matrix[0, 0] * scale
    baseline = np.linalg.norm(t)
    
    depth_map = compute_depth_safely(disparity, focal_length, baseline)
    
    valid_pixels = np.sum(depth_map > 0)
    coverage = (valid_pixels / depth_map.size) * 100
    print(f"Generated depth map: {valid_pixels:,} valid pixels ({coverage:.1f}% coverage)")
    
    return depth_map, img1_small, "SGBM"

def generate_point_cloud_basic(depth_map, camera_matrix, color_image=None):
    """Convert depth map to 3D point cloud"""
    
    print("🔄 Converting depth map to 3D point cloud...")
    
    # Find valid depth pixels
    valid_mask = depth_map > 0
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        print("❌ No valid depth pixels found!")
        return None, None
    
    print(f"📍 Found {valid_count:,} valid depth pixels")
    
    # Get camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Get pixel coordinates
    height, width = depth_map.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Extract valid pixels
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_depths = depth_map[valid_mask]
    
    # Convert to 3D coordinates
    X = (valid_x - cx) * valid_depths / fx
    Y = (valid_y - cy) * valid_depths / fy
    Z = valid_depths
    
    points_3d = np.column_stack((X, Y, Z))
    
    # Handle colors
    colors = None
    if color_image is not None:
        colors = color_image[valid_mask]
    
    print(f"✅ Generated {len(points_3d):,} 3D points")
    
    return points_3d, colors

def visualize_point_cloud_simple(points_3d, colors=None, max_points=50000):
    """Simple 3D visualization of point cloud"""
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ Matplotlib not available. Install with: pip install matplotlib")
        return
    
    print(f"🎨 Visualizing point cloud...")
    
    # Subsample for performance
    if len(points_3d) > max_points:
        print(f"📉 Subsampling {len(points_3d):,} points to {max_points:,}")
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_sub = points_3d[indices]
        colors_sub = colors[indices] if colors is not None else None
    else:
        points_sub = points_3d
        colors_sub = colors
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if colors_sub is not None:
        scatter = ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                           c=colors_sub, cmap='gray', s=1, alpha=0.6)
    else:
        ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                  s=1, alpha=0.6, c='blue')
    
    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (depth, meters)')
    ax.set_title(f'3D Point Cloud\n{len(points_sub):,} points displayed')
    
    print("🎮 Interactive 3D view opened!")
    plt.show()

def generate_basic_mesh(points_3d, colors=None, max_points=20000):
    """Generate triangulated mesh from point cloud"""
    
    print("🔺 === Generating 3D Mesh ===")
    
    if points_3d is None or len(points_3d) == 0:
        print("❌ No points provided")
        return None
    
    print(f"📍 Input: {len(points_3d):,} points")
    
    # Subsample if needed
    if len(points_3d) > max_points:
        print(f"📉 Subsampling to {max_points:,} points")
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_sub = points_3d[indices]
        colors_sub = colors[indices] if colors is not None else None
    else:
        points_sub = points_3d
        colors_sub = colors
    
    try:
        from scipy.spatial import Delaunay
        
        # 2D Delaunay triangulation (project to XY plane)
        points_2d = points_sub[:, :2]
        tri = Delaunay(points_2d)
        
        mesh_data = {
            'vertices': points_sub,
            'faces': tri.simplices,
            'colors': colors_sub,
            'n_vertices': len(points_sub),
            'n_faces': len(tri.simplices),
            'method': 'Delaunay_2D'
        }
        
        print(f"✅ Generated mesh: {mesh_data['n_vertices']:,} vertices, {mesh_data['n_faces']:,} faces")
        return mesh_data
        
    except ImportError:
        print("❌ SciPy not available. Install with: pip install scipy")
        return None
    except Exception as e:
        print(f"❌ Mesh generation failed: {e}")
        return None

def visualize_mesh_simple(mesh_data):
    """Visualize triangulated mesh"""
    
    if mesh_data is None:
        print("❌ No mesh data to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("❌ Matplotlib 3D not available")
        return
    
    print("🎨 Visualizing 3D mesh...")
    
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    colors = mesh_data.get('colors', None)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create triangular faces
    triangles = vertices[faces]
    
    if colors is not None:
        face_colors = np.mean(colors[faces], axis=1)
        face_colors_norm = (face_colors - face_colors.min()) / (face_colors.max() - face_colors.min())
        poly_collection = Poly3DCollection(triangles, alpha=0.7, linewidths=0.5, edgecolors='black')
        poly_collection.set_facecolors(plt.cm.gray(face_colors_norm))
    else:
        poly_collection = Poly3DCollection(triangles, alpha=0.7, linewidths=0.5, 
                                         facecolors='lightblue', edgecolors='black')
    
    ax.add_collection3d(poly_collection)
    
    # Set axis limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (depth, meters)')
    ax.set_title(f'3D Mesh\n{mesh_data["n_vertices"]:,} vertices, {mesh_data["n_faces"]:,} faces')
    
    plt.show()

def multi_view_reconstruction(max_pairs=5):
    """Generate point clouds from multiple image pairs"""
    
    print("🌐 === Multi-View Reconstruction ===")
    
    try:
        K = load_camera_matrix()
    except Exception as e:
        print(f"Error loading camera matrix: {e}")
        return None
    
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    if len(image_files) < 3:
        print(f"Need at least 3 images, found {len(image_files)}")
        return None
    
    print(f"📸 Processing {min(max_pairs, len(image_files)-1)} consecutive pairs")
    
    all_point_clouds = []
    
    for i in range(min(max_pairs, len(image_files)-1)):
        image1 = image_files[i]
        image2 = image_files[i+1]
        
        print(f"\n--- Pair {i+1}: {image1.name} + {image2.name} ---")
        
        try:
            # Feature matching and pose estimation
            result = match_features_between_images(image1, image2)
            if not result:
                print("❌ Feature matching failed")
                continue
                
            matches, kp1, kp2 = result
            R, t = estimate_camera_pose(matches, kp1, kp2, K)
            
            if R is None:
                print("❌ Pose estimation failed")
                continue
            
            # Depth computation
            depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
            
            if depth_map is None:
                print("❌ Depth computation failed")
                continue
            
            # Point cloud generation
            points_3d, colors = generate_point_cloud_basic(depth_map, K, original_img)
            
            if points_3d is None:
                print("❌ Point cloud generation failed")
                continue
            
            valid_points = len(points_3d)
            coverage = (np.sum(depth_map > 0) / depth_map.size) * 100
            
            all_point_clouds.append({
                'points_3d': points_3d,
                'colors': colors,
                'pair_name': f"{image1.name}-{image2.name}",
                'valid_points': valid_points,
                'coverage': coverage
            })
            
            print(f"✅ Success: {valid_points:,} points ({coverage:.1f}% coverage)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Generated {len(all_point_clouds)} point clouds")
    
    if all_point_clouds:
        total_points = sum(pc['valid_points'] for pc in all_point_clouds)
        print(f"Total points: {total_points:,}")
        return all_point_clouds
    
    return None

def merge_point_clouds(point_clouds):
    """Merge multiple point clouds into one"""
    
    print("🔄 Merging point clouds...")
    
    all_points = []
    all_colors = []
    
    for cloud in point_clouds:
        all_points.append(cloud['points_3d'])
        if cloud['colors'] is not None:
            all_colors.append(cloud['colors'])
    
    merged_points = np.vstack(all_points)
    merged_colors = np.concatenate(all_colors) if all_colors else None
    
    print(f"✅ Merged into {len(merged_points):,} total points")
    
    return {
        'points_3d': merged_points,
        'colors': merged_colors,
        'valid_points': len(merged_points),
        'method': 'Merged'
    }

def generate_point_cloud_full_resolution(depth_map, camera_matrix, color_image=None):
    """Generate point cloud using ALL available points (no restrictions)"""
    
    print("🔄 Converting depth map to 3D point cloud (FULL RESOLUTION)...")
    
    # Find ALL valid depth pixels
    valid_mask = depth_map > 0
    valid_count = np.sum(valid_mask)
    
    if valid_count == 0:
        print("❌ No valid depth pixels found!")
        return None, None
    
    print(f"📍 Processing ALL {valid_count:,} valid depth pixels (NO RESTRICTIONS)")
    
    # Get camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Get ALL pixel coordinates
    height, width = depth_map.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Extract ALL valid coordinates (no subsampling)
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_depths = depth_map[valid_mask]
    
    # Convert ALL to 3D coordinates
    X = (valid_x - cx) * valid_depths / fx
    Y = (valid_y - cy) * valid_depths / fy
    Z = valid_depths
    
    points_3d = np.column_stack((X, Y, Z))
    
    # Handle colors for ALL points
    colors = None
    if color_image is not None:
        colors = color_image[valid_mask]
    
    print(f"✅ Generated {len(points_3d):,} 3D points (FULL RESOLUTION - NO LIMITS)")
    
    return points_3d, colors

def visualize_point_cloud_unlimited(points_3d, colors=None, show_all=False):
    """Visualize point cloud with option to show ALL points"""
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ Matplotlib not available")
        return
    
    total_points = len(points_3d)
    print(f"🎨 Visualizing point cloud with {total_points:,} total points...")
    
    if show_all or total_points <= 100000:
        # Show ALL points (or reasonable amount)
        points_to_show = points_3d
        colors_to_show = colors
        print(f"📊 Showing ALL {len(points_to_show):,} points")
    else:
        # Subsample only if really necessary
        max_points = 100000
        print(f"📉 Subsampling {total_points:,} points to {max_points:,} for performance")
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_to_show = points_3d[indices]
        colors_to_show = colors[indices] if colors is not None else None
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))  # Larger figure
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with smaller point size for better detail
    if colors_to_show is not None:
        scatter = ax.scatter(points_to_show[:, 0], points_to_show[:, 1], points_to_show[:, 2], 
                           c=colors_to_show, cmap='gray', s=0.5, alpha=0.8)  # Smaller points
    else:
        ax.scatter(points_to_show[:, 0], points_to_show[:, 1], points_to_show[:, 2], 
                  s=0.5, alpha=0.8, c='blue')
    
    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (depth, meters)')
    ax.set_title(f'High-Resolution 3D Point Cloud\n{len(points_to_show):,} points displayed')
    
    print("🎮 High-resolution 3D view opened!")
    print("   Try rotating and zooming to see fine details!")
    plt.show()

def analyze_point_cloud_structure(points_3d, colors=None):
    """Analyze the structure and distribution of the point cloud"""
    
    print("🔍 === Point Cloud Analysis ===")
    
    if points_3d is None or len(points_3d) == 0:
        print("❌ No points to analyze")
        return
    
    print(f"📊 Total points: {len(points_3d):,}")
    
    # Spatial distribution analysis
    x_range = points_3d[:, 0].max() - points_3d[:, 0].min()
    y_range = points_3d[:, 1].max() - points_3d[:, 1].min()
    z_range = points_3d[:, 2].max() - points_3d[:, 2].min()
    
    print(f"📏 Spatial dimensions:")
    print(f"   X range: {x_range:.2f} meters")
    print(f"   Y range: {y_range:.2f} meters") 
    print(f"   Z range: {z_range:.2f} meters")
    
    # Density analysis
    volume_estimate = x_range * y_range * z_range
    density = len(points_3d) / volume_estimate if volume_estimate > 0 else 0
    print(f"📈 Point density: ~{density:.1f} points per cubic meter")
    
    # Height distribution (assuming statue is vertical)
    y_coords = points_3d[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    statue_height_estimate = y_max - y_min
    print(f"🏛️ Estimated statue height: {statue_height_estimate:.2f} meters")
    
    # Analyze depth layers (Z direction)
    z_coords = points_3d[:, 2]
    z_layers = np.linspace(z_coords.min(), z_coords.max(), 10)
    
    print(f"📋 Point distribution by depth layers:")
    for i in range(len(z_layers)-1):
        layer_mask = (z_coords >= z_layers[i]) & (z_coords < z_layers[i+1])
        layer_count = np.sum(layer_mask)
        layer_percent = (layer_count / len(points_3d)) * 100
        print(f"   Layer {i+1}: {layer_count:,} points ({layer_percent:.1f}%)")
    
    # Color analysis if available
    if colors is not None:
        color_range = colors.max() - colors.min()
        color_mean = colors.mean()
        print(f"🎨 Color analysis:")
        print(f"   Range: {color_range:.1f}")
        print(f"   Mean: {color_mean:.1f}")
    
    return {
        'total_points': len(points_3d),
        'spatial_range': (x_range, y_range, z_range),
        'density': density,
        'height_estimate': statue_height_estimate
    }

def visualize_point_cloud_with_analysis(points_3d, colors=None):
    """Enhanced visualization with analysis overlays"""
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ Matplotlib not available")
        return
    
    # Analyze first
    analysis = analyze_point_cloud_structure(points_3d, colors)
    
    # Create enhanced visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Subsample for performance if needed
    if len(points_3d) > 100000:
        indices = np.random.choice(len(points_3d), 100000, replace=False)
        points_sub = points_3d[indices]
        colors_sub = colors[indices] if colors is not None else None
    else:
        points_sub = points_3d
        colors_sub = colors
    
    if colors_sub is not None:
        scatter = ax1.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                            c=colors_sub, cmap='gray', s=0.5, alpha=0.7)
    else:
        ax1.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2], 
                   s=0.5, alpha=0.7, c='blue')
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_zlabel('Z (depth, meters)')
    ax1.set_title(f'3D Point Cloud\n{analysis["total_points"]:,} points')
    
    # Top view (X-Z plane)
    ax2 = fig.add_subplot(222)
    ax2.scatter(points_sub[:, 0], points_sub[:, 2], s=0.1, alpha=0.5)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (depth, meters)')
    ax2.set_title('Top View (X-Z)')
    ax2.grid(True)
    
    # Side view (Y-Z plane)
    ax3 = fig.add_subplot(223)
    ax3.scatter(points_sub[:, 1], points_sub[:, 2], s=0.1, alpha=0.5)
    ax3.set_xlabel('Y (meters)')
    ax3.set_ylabel('Z (depth, meters)')
    ax3.set_title('Side View (Y-Z)')
    ax3.grid(True)
    
    # Front view (X-Y plane)
    ax4 = fig.add_subplot(224)
    ax4.scatter(points_sub[:, 0], points_sub[:, 1], s=0.1, alpha=0.5)
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    ax4.set_title('Front View (X-Y)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return analysis

def multi_view_reconstruction_dense(max_pairs=8):
    """Multi-view reconstruction using FULL resolution point clouds"""
    
    print("🌐 === Dense Multi-View Reconstruction ===")
    print("Using FULL resolution for maximum detail")
    
    try:
        K = load_camera_matrix()
    except Exception as e:
        print(f"Error loading camera matrix: {e}")
        return None
    
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    if len(image_files) < 3:
        print(f"Need at least 3 images, found {len(image_files)}")
        return None
    
    print(f"📸 Processing {min(max_pairs, len(image_files)-1)} pairs with FULL resolution")
    
    all_point_clouds = []
    
    for i in range(min(max_pairs, len(image_files)-1)):
        image1 = image_files[i]
        image2 = image_files[i+1]
        
        print(f"\n--- Dense Pair {i+1}: {image1.name} + {image2.name} ---")
        
        try:
            result = match_features_between_images(image1, image2)
            if not result:
                print("❌ Feature matching failed")
                continue
                
            matches, kp1, kp2 = result
            R, t = estimate_camera_pose(matches, kp1, kp2, K)
            
            if R is None:
                print("❌ Pose estimation failed")
                continue
            
            depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
            
            if depth_map is None:
                print("❌ Depth computation failed")
                continue
            
            # Use FULL RESOLUTION point cloud generation
            points_3d, colors = generate_point_cloud_full_resolution(depth_map, K, original_img)
            
            if points_3d is None:
                print("❌ Point cloud generation failed")
                continue
            
            valid_points = len(points_3d)
            coverage = (np.sum(depth_map > 0) / depth_map.size) * 100
            
            all_point_clouds.append({
                'points_3d': points_3d,
                'colors': colors,
                'pair_name': f"{image1.name}-{image2.name}",
                'valid_points': valid_points,
                'coverage': coverage
            })
            
            print(f"✅ Success: {valid_points:,} points ({coverage:.1f}% coverage)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Generated {len(all_point_clouds)} dense point clouds")
    
    if all_point_clouds:
        total_points = sum(pc['valid_points'] for pc in all_point_clouds)
        print(f"Total points across all reconstructions: {total_points:,}")
        return all_point_clouds
    
    return None

def export_point_cloud(points_3d, colors=None, filename="statue_reconstruction"):
    """Export point cloud to PLY format for external viewing"""
    
    print(f"💾 Exporting point cloud to {filename}.ply...")
    
    with open(f"{filename}.ply", 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertices
        for i, point in enumerate(points_3d):
            if colors is not None:
                color = int(colors[i])
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color} {color} {color}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"✅ Exported {len(points_3d):,} points to {filename}.ply")
    print("   📂 You can open this file in:")
    print("      - MeshLab (free)")
    print("      - CloudCompare (free)")
    print("      - Blender (free)")
    print("      - Any 3D software that supports PLY format")

if __name__ == "__main__":
    print("🌊 === 3D Reconstruction Pipeline ===")
    
    # Validate setup
    print("\n🔍 Checking data setup...")
    issues = validate_data_setup()
    if issues:
        print("❌ Setup issues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    
    print("✅ Data setup looks good!")

    print("\nChoose reconstruction mode:")
    print("1. Single pair reconstruction")
    print("2. Point cloud generation") 
    print("3. Multi-view reconstruction")
    print("4. Generate 3D mesh")
    print("5. Full pipeline (multi-view + merge)")
    print("6. UNRESTRICTED processing (use ALL features & points!)")
    print("7. Analyze point cloud structure")
    print("8. Dense multi-view reconstruction")
    print("9. 3D Visualization Only (from existing reconstruction)")
    print("10. Export point cloud to file")
    print("11. Complete 3D pipeline with visualization")
    
    choice = input("Enter choice (1-11): ").strip()

    # Load camera matrix
    try:
        K = load_camera_matrix()
        print("✅ Camera matrix loaded")
    except Exception as e:
        print(f"❌ Error loading camera matrix: {e}")
        sys.exit(1)
    
    if choice == "1":
        # Single pair reconstruction
        print("\n--- Single Pair Reconstruction ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            print("✅ Reconstruction successful!")
                            
                            # Show basic statistics
                            valid_pixels = np.sum(depth_map > 0)
                            coverage = (valid_pixels / depth_map.size) * 100
                            print(f"Valid pixels: {valid_pixels:,} ({coverage:.1f}% coverage)")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")
    
    elif choice == "2":
        # Point cloud generation
        print("\n--- Point Cloud Generation ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            points_3d, colors = generate_point_cloud_basic(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                print("🎉 Point cloud generated!")
                                visualize_point_cloud_simple(points_3d, colors)
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")
    
    elif choice == "3":
        # Multi-view reconstruction
        print("\n--- Multi-View Reconstruction ---")
        try:
            point_clouds = multi_view_reconstruction(max_pairs=5)
            
            if point_clouds:
                print("🎉 Multi-view reconstruction complete!")
                
                # Show best result
                best_cloud = max(point_clouds, key=lambda x: x['coverage'])
                print(f"Best result: {best_cloud['pair_name']} ({best_cloud['coverage']:.1f}% coverage)")
                visualize_point_cloud_simple(best_cloud['points_3d'], best_cloud['colors'])
            else:
                print("❌ Multi-view reconstruction failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "4":
        # Generate mesh
        print("\n--- 3D Mesh Generation ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            points_3d, colors = generate_point_cloud_basic(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                mesh_data = generate_basic_mesh(points_3d, colors)
                                
                                if mesh_data is not None:
                                    print("🎉 Mesh generated!")
                                    visualize_mesh_simple(mesh_data)
                                else:
                                    print("❌ Mesh generation failed")
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")
    
    elif choice == "5":
        # Full pipeline
        print("\n--- Full Pipeline (Multi-view + Merge) ---")
        try:
            point_clouds = multi_view_reconstruction(max_pairs=8)
            
            if point_clouds and len(point_clouds) > 1:
                print("🔄 Merging all point clouds...")
                merged_cloud = merge_point_clouds(point_clouds)
                
                print("🎉 Full pipeline complete!")
                print(f"Final model: {merged_cloud['valid_points']:,} points")
                visualize_point_cloud_simple(merged_cloud['points_3d'], merged_cloud['colors'], max_points=100000)
                
            else:
                print("❌ Need multiple point clouds to merge")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif choice == "6":
        # Unrestricted processing
        print("\n--- UNRESTRICTED PROCESSING ---")
        print("🚀 Using ALL features and ALL points (may be slow but more detailed)")
        
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    print(f"✅ Found {len(matches)} feature matches (unrestricted)")
                    
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            # Use FULL RESOLUTION point cloud generation
                            points_3d, colors = generate_point_cloud_full_resolution(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                print("🎉 Unrestricted point cloud generated!")
                                print(f"Total points: {len(points_3d):,}")
                                
                                # Ask user if they want to see all points
                                show_all = input("Show ALL points? (may be slow) (y/n): ").strip().lower() == 'y'
                                visualize_point_cloud_unlimited(points_3d, colors, show_all=show_all)
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")
    elif choice == "7":
        # Point cloud analysis
        print("\n--- Point Cloud Analysis ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            points_3d, colors = generate_point_cloud_full_resolution(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                print("🎉 Analyzing dense point cloud...")
                                analysis = visualize_point_cloud_with_analysis(points_3d, colors)
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")
    
    elif choice == "8":
        # Dense multi-view reconstruction
        print("\n--- Dense Multi-View Reconstruction ---")
        try:
            point_clouds = multi_view_reconstruction_dense(max_pairs=5)
            
            if point_clouds:
                print("🎉 Dense multi-view reconstruction complete!")
                
                # Merge all for complete model
                merged_cloud = merge_point_clouds(point_clouds)
                print(f"🔥 COMPLETE DENSE MODEL: {merged_cloud['valid_points']:,} points")
                
                # Enhanced visualization
                print("Opening enhanced visualization...")
                visualize_point_cloud_with_analysis(merged_cloud['points_3d'], merged_cloud['colors'])
            else:
                print("❌ Dense multi-view reconstruction failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif choice == "9":
        # 3D Visualization only
        print("\n--- 3D Visualization Mode ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            points_3d, colors = generate_point_cloud_full_resolution(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                print("🎉 Opening 3D visualization...")
                                
                                # Give user visualization options
                                print("\nVisualization options:")
                                print("a) Standard 3D view")
                                print("b) High-resolution 3D view") 
                                print("c) Multi-panel analysis view")
                                
                                viz_choice = input("Choose visualization (a/b/c): ").strip().lower()
                                
                                if viz_choice == 'a':
                                    visualize_point_cloud_simple(points_3d, colors, max_points=50000)
                                elif viz_choice == 'b':
                                    visualize_point_cloud_unlimited(points_3d, colors, show_all=False)
                                elif viz_choice == 'c':
                                    visualize_point_cloud_with_analysis(points_3d, colors)
                                else:
                                    print("Default: Standard 3D view")
                                    visualize_point_cloud_simple(points_3d, colors)
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")

    elif choice == "10":
        # Export point cloud
        print("\n--- Export Point Cloud ---")
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1, image2 = image_files[0], image_files[1]
            
            try:
                result = match_features_between_images(image1, image2)
                if result:
                    matches, kp1, kp2 = result
                    R, t = estimate_camera_pose(matches, kp1, kp2, K)
                    
                    if R is not None:
                        depth_map, original_img, method = compute_improved_depth_map(image1, image2, K, R, t)
                        
                        if depth_map is not None:
                            points_3d, colors = generate_point_cloud_full_resolution(depth_map, K, original_img)
                            
                            if points_3d is not None:
                                export_point_cloud(points_3d, colors, "statue_reconstruction")
                            else:
                                print("❌ Point cloud generation failed")
                        else:
                            print("❌ Depth computation failed")
                    else:
                        print("❌ Pose estimation failed")
                else:
                    print("❌ Feature matching failed")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Need at least 2 images")

    elif choice == "11":
        # Complete 3D pipeline with visualization
        print("\n--- Complete 3D Pipeline with Visualization ---")
        print("🚀 This will create the most complete 3D model and show it!")
        
        try:
            # Step 1: Multi-view reconstruction
            print("\n🔄 Step 1: Multi-view reconstruction...")
            point_clouds = multi_view_reconstruction_dense(max_pairs=6)
            
            if point_clouds and len(point_clouds) > 1:
                # Step 2: Merge point clouds
                print("\n🔄 Step 2: Merging point clouds...")
                merged_cloud = merge_point_clouds(point_clouds)
                
                print(f"🎉 Complete 3D model created!")
                print(f"📊 Final statistics:")
                print(f"   - Total points: {merged_cloud['valid_points']:,}")
                print(f"   - Source reconstructions: {len(point_clouds)}")
                
                # Step 3: Multiple visualization options
                print("\n🎨 Step 3: 3D Visualization...")
                print("Choose visualization type:")
                print("a) Interactive 3D point cloud")
                print("b) Multi-panel analysis view")
                print("c) High-resolution 3D view")
                print("d) 3D mesh generation")
                print("e) All visualizations")
                
                viz_choice = input("Choose visualization (a/b/c/d/e): ").strip().lower()
                
                if viz_choice == 'a' or viz_choice == 'e':
                    print("📊 Opening interactive 3D point cloud...")
                    visualize_point_cloud_simple(merged_cloud['points_3d'], merged_cloud['colors'], max_points=100000)
                
                if viz_choice == 'b' or viz_choice == 'e':
                    print("📈 Opening multi-panel analysis...")
                    visualize_point_cloud_with_analysis(merged_cloud['points_3d'], merged_cloud['colors'])
                
                if viz_choice == 'c' or viz_choice == 'e':
                    print("🔍 Opening high-resolution view...")
                    visualize_point_cloud_unlimited(merged_cloud['points_3d'], merged_cloud['colors'], show_all=False)
                
                if viz_choice == 'd' or viz_choice == 'e':
                    print("🔺 Generating 3D mesh...")
                    mesh_data = generate_basic_mesh(merged_cloud['points_3d'], merged_cloud['colors'], max_points=30000)
                    if mesh_data:
                        print("🎨 Opening 3D mesh visualization...")
                        visualize_mesh_simple(mesh_data)
                
                # Step 4: Export option
                export_option = input("\nExport point cloud to file? (y/n): ").strip().lower()
                if export_option == 'y':
                    export_point_cloud(merged_cloud['points_3d'], merged_cloud['colors'], "complete_statue_model")
                
                print("\n🎉 Complete 3D reconstruction pipeline finished!")
                
            else:
                print("❌ Need multiple point clouds for complete pipeline")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Invalid choice")