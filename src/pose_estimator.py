import cv2
import numpy as np
from pathlib import Path
from feature_matcher import match_features_between_images
from config import DATASET_NAME

def estimate_camera_pose(matches, kp1, kp2, camera_matrix):
    """Estimate camera pose between two images"""
    
    if len(matches) < 8:
        print("Need at least 8 matches for pose estimation")
        return None, None
    
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    print(f"Using {len(matches)} matches for pose estimation")
    
    # Find Essential Matrix
    essential_matrix, mask = cv2.findEssentialMat(
        pts1, pts2, camera_matrix, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0
    )
    
    # Recover rotation and translation
    _, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)
    
    print(f"Pose estimation successful!")
    print(f"Rotation matrix shape: {R.shape}")
    print(f"Translation vector shape: {t.shape}")
    
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
                R, t = estimate_camera_pose(matches, kp1, kp2, K)
                
                if R is not None:
                    print(f"\n✅ Camera motion estimated!")
                    print(f"Translation (camera movement): {t.flatten()}")
                    print(f"Camera moved: {np.linalg.norm(t):.3f} units")