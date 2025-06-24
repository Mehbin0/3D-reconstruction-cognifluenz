import numpy as np
from pathlib import Path
from config import DATASET_NAME

# Import all our custom modules
from feature_matcher import match_features_between_images
from pose_estimator import load_camera_matrix, estimate_camera_pose
from triangulation import triangulate_points, analyze_reconstruction

def reconstruct_statue_3d():
    """Complete 3D reconstruction pipeline"""
    
    print("🗿 === Statue 3D Reconstruction Pipeline ===")
    
    # Step 1: Load camera calibration
    print("\n📷 Loading camera parameters...")
    K = load_camera_matrix()
    if K is None:
        print("❌ Failed to load camera matrix")
        return None
    
    # Step 2: Get image pairs
    print("\n🖼️ Loading image pairs...")
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    if len(image_files) < 2:
        print("❌ Need at least 2 images")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Step 3: Process first image pair
    image1, image2 = image_files[0], image_files[1]
    print(f"Processing: {image1.name} + {image2.name}")
    
    # Step 4: Extract and match features
    print("\n🔍 Matching features...")
    result = match_features_between_images(image1, image2)
    if not result:
        print("❌ Feature matching failed")
        return None
    
    matches, kp1, kp2 = result
    
    # Step 5: Estimate camera pose
    print("\n📐 Estimating camera pose...")
    R, t = estimate_camera_pose(matches, kp1, kp2, K)
    if R is None:
        print("❌ Pose estimation failed")
        return None
    
    # Step 6: Triangulate 3D points
    print("\n🌐 Triangulating 3D points...")
    points_3d = triangulate_points(matches, kp1, kp2, K, R, t)
    
    if points_3d is not None and len(points_3d) > 0:
        print(f"\n🎉 Reconstruction successful!")
        analyze_reconstruction(points_3d)
        
        # Save results
        save_reconstruction(points_3d, "statue_3d_points.txt")
        
        return points_3d
    else:
        print("❌ Triangulation failed")
        return None

def save_reconstruction(points_3d, filename):
    """Save 3D points to file"""
    output_path = Path(f"../data/{DATASET_NAME}/{filename}")
    
    with open(output_path, 'w') as f:
        f.write("# Statue 3D Reconstruction\n")
        f.write("# X Y Z coordinates\n")
        for i, point in enumerate(points_3d):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"💾 Saved reconstruction to: {output_path}")

def reconstruct_multiple_pairs(max_pairs=5):
    """Reconstruct 3D points from multiple image pairs"""
    
    print("🌐 === Multi-View 3D Reconstruction ===")
    
    # Load camera parameters
    K = load_camera_matrix()
    if K is None:
        return None
    
    # Get all images
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    print(f"Found {len(image_files)} images")
    print(f"Will process {min(max_pairs, len(image_files)-1)} consecutive pairs")
    
    all_points_3d = []
    successful_pairs = 0
    
    # Process consecutive image pairs
    for i in range(min(max_pairs, len(image_files)-1)):
        image1 = image_files[i]
        image2 = image_files[i+1]
        
        print(f"\n--- Pair {i+1}: {image1.name} + {image2.name} ---")
        
        # Match features
        result = match_features_between_images(image1, image2)
        if not result:
            print("❌ Feature matching failed")
            continue
        
        matches, kp1, kp2 = result
        
        # Estimate pose
        R, t = estimate_camera_pose(matches, kp1, kp2, K)
        if R is None:
            print("❌ Pose estimation failed")
            continue
        
        # Triangulate
        points_3d = triangulate_points(matches, kp1, kp2, K, R, t)
        
        if points_3d is not None and len(points_3d) > 0:
            all_points_3d.append(points_3d)
            successful_pairs += 1
            print(f"✅ Success: {len(points_3d)} points")
        else:
            print("❌ Triangulation failed")
    
    if all_points_3d:
        combined_points = np.vstack(all_points_3d)
        
        print(f"\n🎉 Multi-view reconstruction complete!")
        print(f"Total raw 3D points: {len(combined_points)}")
        
        # Analyze different thresholds
        analyze_deduplication_thresholds(combined_points)

        # Remove duplicates
        unique_points = remove_duplicate_points_simple(combined_points, distance_threshold=0.3)
        
        # Analyze final result
        analyze_reconstruction(unique_points)
        save_reconstruction(unique_points, "statue_unique_3d.txt")
        
        return unique_points
    
    return None

def remove_duplicate_points_simple(points_3d, distance_threshold=0.5):
    """Simple duplicate removal without external libraries"""
    
    if len(points_3d) == 0:
        return points_3d
    
    print(f"\n🔧 Removing duplicates from {len(points_3d)} points...")
    
    unique_points = []
    
    for point in points_3d:
        # Check if this point is too close to any existing unique point
        is_duplicate = False
        for unique_point in unique_points:
            distance = np.linalg.norm(point - unique_point)
            if distance < distance_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_points.append(point)
    
    unique_points = np.array(unique_points)
    
    print(f"✅ Reduced to {len(unique_points)} unique points")
    print(f"Removed {len(points_3d) - len(unique_points)} duplicates")
    
    return unique_points

def analyze_deduplication_thresholds(points_3d):
    """Test different distance thresholds"""
    
    print(f"\n🔍 === Deduplication Analysis ===")
    print(f"Starting with {len(points_3d)} points")
    
    thresholds = [0.1, 0.3, 0.5, 1.0, 2.0]
    
    for threshold in thresholds:
        unique_points = remove_duplicate_points_simple(points_3d, threshold)
        reduction_percent = (1 - len(unique_points)/len(points_3d)) * 100
        print(f"Threshold {threshold}: {len(unique_points)} points ({reduction_percent:.1f}% reduction)")
    
    print(f"\nRecommendation: Try threshold 0.3-0.5 for balance")

def reconstruct_all_pairs(max_pairs=15):
    """Try different image pair combinations with various spacing"""
    
    print("🚀 === Advanced Multi-View Reconstruction ===")
    
    K = load_camera_matrix()
    if K is None:
        return None
    
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    print(f"Found {len(image_files)} images")
    
    all_points_3d = []
    pair_count = 0
    successful_pairs = 0
    
    # Try pairs with different spacing
    spacings = [1, 2, 3, 4]  # consecutive, skip 1, skip 2, skip 3
    
    for spacing in spacings:
        print(f"\n--- Testing spacing {spacing} (every {spacing+1} images) ---")
        
        for i in range(len(image_files) - spacing):
            if pair_count >= max_pairs:
                break
                
            image1 = image_files[i]
            image2 = image_files[i + spacing]
            
            print(f"Pair {pair_count + 1}: {image1.name} + {image2.name}")
            
            # Reconstruction pipeline
            result = match_features_between_images(image1, image2)
            if not result:
                print("  ❌ Feature matching failed")
                pair_count += 1
                continue
            
            matches, kp1, kp2 = result
            
            R, t = estimate_camera_pose(matches, kp1, kp2, K)
            if R is None:
                print("  ❌ Pose estimation failed")
                pair_count += 1
                continue
            
            points_3d = triangulate_points(matches, kp1, kp2, K, R, t)
            
            if points_3d is not None and len(points_3d) > 0:
                all_points_3d.append(points_3d)
                successful_pairs += 1
                print(f"  ✅ Success: {len(points_3d)} points")
            else:
                print("  ❌ Triangulation failed")
            
            pair_count += 1
        
        if pair_count >= max_pairs:
            break
    
    print(f"\n📊 Processing summary:")
    print(f"Total pairs attempted: {pair_count}")
    print(f"Successful pairs: {successful_pairs}")
    
    if all_points_3d:
        combined_points = np.vstack(all_points_3d)
        
        print(f"Total raw 3D points: {len(combined_points)}")
        
        # Analyze thresholds
        analyze_deduplication_thresholds(combined_points)
        
        # Remove duplicates with optimal threshold
        unique_points = remove_duplicate_points_simple(combined_points, distance_threshold=0.3)
        
        # Analyze and save
        analyze_reconstruction(unique_points)
        save_reconstruction(unique_points, "statue_advanced_3d.txt")
        
        return unique_points
    
    return None

if __name__ == "__main__":
    print("Choose reconstruction mode:")
    print("1. Single pair (quick test)")
    print("2. Multiple consecutive pairs")
    print("3. Advanced multi-spacing pairs")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        points_3d = reconstruct_statue_3d()
    elif choice == "2":
        points_3d = reconstruct_multiple_pairs(max_pairs=5)
    elif choice == "3":
        points_3d = reconstruct_all_pairs(max_pairs=15)
    else:
        print("Invalid choice")
        points_3d = None
    
    if points_3d is not None:
        print(f"\n✨ Final result: {len(points_3d)} 3D points!")
        print("🎯 Your 3D reconstruction is complete!")