import cv2
import numpy as np
from pathlib import Path

def extract_orb_features(image_path):
    """Extract ORB features from a single image"""
    
    # Step 1: Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Step 2: Convert to grayscale (ORB works on grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)  # Find up to 1000 features
    
    # Step 4: Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    print(f"Found {len(keypoints)} features in {image_path.name}")
    
    return keypoints, descriptors

def visualize_features(image_path, keypoints):
    """Draw the detected features on the image"""
    
    # Load the image
    image = cv2.imread(str(image_path))
    
    # Draw keypoints on the image
    image_with_features = cv2.drawKeypoints(
        image, keypoints, None, 
        color=(0, 255, 0),  # Green color
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return image_with_features

def compare_orb_settings(image_path):
    """Compare different ORB parameter settings"""
    
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Test different settings
    settings = [
        {"nfeatures": 500, "name": "Conservative"},
        {"nfeatures": 1000, "name": "Default"},
        {"nfeatures": 2000, "name": "More features"},
    ]
    
    for setting in settings:
        orb = cv2.ORB_create(nfeatures=setting["nfeatures"])
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        print(f"{setting['name']} (nfeatures={setting['nfeatures']}): Found {len(keypoints)} features")

def analyze_feature_quality(image_path):
    """Analyze the quality/strength of detected features"""
    
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Test with different feature counts
    for nfeatures in [500, 1000, 2000]:
        orb = cv2.ORB_create(nfeatures=nfeatures)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Get the response (strength) of each keypoint
        responses = [kp.response for kp in keypoints]
        if responses:
            avg_response = sum(responses) / len(responses)
            min_response = min(responses)
            max_response = max(responses)
            
            print(f"nfeatures={nfeatures}: Found {len(keypoints)} features")
            print(f"  Average strength: {avg_response:.6f}")  # More decimal places
            print(f"  Weakest feature: {min_response:.6f}")
            print(f"  Strongest feature: {max_response:.6f}")
            print()

if __name__ == "__main__":
    # Test with one image from our statue dataset
    images_path = Path("../data/statue/images/dslr_images_undistorted")
    
    # Get the first image file
    image_files = list(images_path.glob("*.JPG"))
    if image_files:
        first_image = image_files[0]
        print(f"Testing feature extraction on: {first_image.name}")
        
        # Extract features
        keypoints, descriptors = extract_orb_features(first_image)
        
        if keypoints is not None:
            print(f"Success! Found {len(keypoints)} features")
            
            # Visualize the features
            image_with_features = visualize_features(first_image, keypoints)
            
            # Save the result so you can see it
            output_path = Path("../data/statue/features_visualization.jpg")
            cv2.imwrite(str(output_path), image_with_features)
            print(f"Saved visualization to: {output_path}")
        else:
            print("Feature extraction failed")
    else:
        print("No images found!")
    
    # Compare different ORB settings
    print("\n=== Comparing ORB Settings ===")
    compare_orb_settings(first_image)

    print("\n=== Feature Quality Analysis ===")
    analyze_feature_quality(first_image)