import cv2
import numpy as np
from pathlib import Path
from feature_matcher import match_features_between_images
from config import DATASET_NAME

# Estimate camera pose
def estimate_pose(matches, camera_matrix):
    pts1 = np.float32([m.queryIdx for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([m.trainIdx for m in matches]).reshape(-1, 1, 2)
    essential_matrix, _ = cv2.findEssentialMat(pts1, pts2, camera_matrix)
    _, R, t, _ = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)
    return R, t

def load_camera_matrix():
    """Load camera parameters from ETH3D data"""
    
    cameras_file = Path(f"../data/{DATASET_NAME}/calibration/cameras.txt")
    
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    # Find the camera data (skip comments)
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            
            # Extract camera parameters
            fx = float(parts[4])
            fy = float(parts[5]) 
            cx = float(parts[6])
            cy = float(parts[7])
            
            # Create intrinsic matrix
            camera_matrix = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ])
            
            print("Camera Intrinsic Matrix:")
            print(camera_matrix)
            
            return camera_matrix
    
    return None

if __name__ == "__main__":
    print("=== Loading Camera Matrix ===")
    K = load_camera_matrix()
    
    if K is not None:
        print("\n=== Testing Pose Estimation ===")
        
        # Import our feature matching
        from feature_matcher import match_features_between_images
        
        # Get two consecutive images
        images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
        image_files = sorted(list(images_path.glob("*.JPG")))
        
        if len(image_files) >= 2:
            image1 = image_files[0]
            image2 = image_files[1]
            
            # Get matches
            result = match_features_between_images(image1, image2)
            
            if result:
                matches, kp1, kp2 = result
                
                # Estimate pose
                R, t = estimate_pose(matches, K)
                
                if R is not None:
                    print(f"\n✅ Camera motion estimated!")
                    print(f"Translation (camera movement): {t.flatten()}")
                    print(f"Camera moved: {np.linalg.norm(t):.3f} units")
    
    print("Pose estimation completed successfully")