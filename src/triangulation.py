import cv2
import numpy as np
from pathlib import Path
from config import DATASET_NAME

def triangulate_points(matches, kp1, kp2, camera_matrix, R, t):
    """Create 3D points from matched features - FIXED VERSION"""
    
    if len(matches) < 8:
        print("Need at least 8 matches for triangulation")
        return None
    
    # Extract point coordinates properly
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    
    print(f"Input points: {len(pts1)} matches")
    
    # Apply geometric filtering (fundamental matrix)
    F, mask1 = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        print("Could not find fundamental matrix")
        return None
    
    # Keep only geometrically consistent points
    pts1_good = pts1[mask1.ravel() == 1]
    pts2_good = pts2[mask1.ravel() == 1]
    
    print(f"After fundamental matrix filtering: {len(pts1_good)} points")
    
    if len(pts1_good) < 8:
        print("Not enough points after geometric filtering")
        return None
    
    # Verify with essential matrix
    E, mask2 = cv2.findEssentialMat(pts1_good, pts2_good, camera_matrix, method=cv2.RANSAC)
    if E is None:
        print("Could not find essential matrix")
        return None
    
    # Recover pose (this gives us the final good points)
    _, R_recovered, t_recovered, mask3 = cv2.recoverPose(E, pts1_good, pts2_good, camera_matrix)
    
    # Keep only the final verified points
    final_pts1 = pts1_good[mask3.ravel() > 0]
    final_pts2 = pts2_good[mask3.ravel() > 0]
    
    print(f"Final verified points for triangulation: {len(final_pts1)}")
    
    if len(final_pts1) < 4:
        print("Not enough final points for triangulation")
        return None
    
    # Create projection matrices correctly
    P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera_matrix @ np.hstack([R, t])
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, final_pts1.T, final_pts2.T)
    
    # Convert to 3D with proper validation
    points_3d = []
    for i in range(points_4d.shape[1]):
        w = points_4d[3, i]
        if abs(w) > 1e-8:  # Valid point (not at infinity)
            x = float(points_4d[0, i] / w)
            y = float(points_4d[1, i] / w) 
            z = float(points_4d[2, i] / w)
            
            # Basic depth check (points should be in front of camera)
            if z > 0:
                points_3d.append([x, y, z])
    
    points_3d = np.array(points_3d)
    
    print(f"✅ Successfully triangulated {len(points_3d)} valid 3D points!")
    
    return points_3d

def analyze_reconstruction(points_3d):
    """Analyze the 3D reconstruction results"""
    
    if points_3d is None or len(points_3d) == 0:
        print("No 3D points to analyze")
        return
    
    print(f"\n=== 3D Reconstruction Analysis ===")
    print(f"Total 3D points: {len(points_3d)}")
    
    # Calculate statistics
    min_coords = np.min(points_3d, axis=0)
    max_coords = np.max(points_3d, axis=0)
    mean_coords = np.mean(points_3d, axis=0)
    
    print(f"Bounding box:")
    print(f"  X: {min_coords[0]:.2f} to {max_coords[0]:.2f}")
    print(f"  Y: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
    print(f"  Z: {min_coords[2]:.2f} to {max_coords[2]:.2f}")
    
    print(f"Center point: ({mean_coords[0]:.2f}, {mean_coords[1]:.2f}, {mean_coords[2]:.2f})")
    
    # Show sample points
    print(f"\nSample 3D points:")
    for i in range(min(5, len(points_3d))):
        x, y, z = points_3d[i]
        print(f"  Point {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")

if __name__ == "__main__":
    print("=== Fixed 3D Reconstruction Test ===")
    
    # Import everything we need
    from pose_estimator import load_camera_matrix, estimate_camera_pose
    from feature_matcher import match_features_between_images
    
    # Load camera parameters
    K = load_camera_matrix()
    
    if K is not None:
        # Get two images
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1 = image_files[0]
            image2 = image_files[1]
            
            print(f"\nReconstructing from:")
            print(f"  {image1.name} and {image2.name}")
            
            # Step 1: Match features
            result = match_features_between_images(image1, image2)
            
            if result:
                matches, kp1, kp2 = result
                
                # Step 2: Estimate camera pose
                R, t = estimate_camera_pose(matches, kp1, kp2, K)
                
                if R is not None:
                    # Step 3: Triangulate 3D points (FIXED)
                    points_3d = triangulate_points(matches, kp1, kp2, K, R, t)
                    
                    # Step 4: Analyze results
                    analyze_reconstruction(points_3d)