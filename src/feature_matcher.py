import cv2
import numpy as np
from pathlib import Path
from feature_extractor import extract_orb_features

def match_features_between_images(image1_path, image2_path):
    """Find matching features between two images"""
    
    print(f"Matching features between:")
    print(f"  Image 1: {image1_path.name}")
    print(f"  Image 2: {image2_path.name}")
    
    # Step 1: Extract features from both images
    kp1, desc1 = extract_orb_features(image1_path)
    kp2, desc2 = extract_orb_features(image2_path)
    
    if desc1 is None or desc2 is None:
        print("Error: Could not extract features from one or both images")
        return None
    
    print(f"Image 1 has {len(kp1)} features")
    print(f"Image 2 has {len(kp2)} features")
    
    # Step 2 & 3: Find and filter matches
    good_matches = filter_matches_with_ratio_test(desc1, desc2)

    print(f"Found {len(good_matches)} good matches after filtering")

    return good_matches, kp1, kp2

def visualize_matches(image1_path, image2_path, matches, kp1, kp2):
    """Draw the matches between two images"""
    
    # Load both images
    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    
    # Draw matches
    matched_image = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:50],  # Show first 50 matches
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return matched_image

def filter_matches_with_ratio_test(desc1, desc2, ratio_threshold=0.75):
    """Apply Lowe's ratio test to filter good matches"""
    
    # Use KNN matcher to get 2 best matches for each feature
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:  # Make sure we have 2 matches
            best_match, second_best = match_pair
            
            # If best match is significantly better than second best
            if best_match.distance < ratio_threshold * second_best.distance:
                good_matches.append(best_match)
    
    return good_matches

if __name__ == "__main__":
    # Test with two consecutive images from our statue
    images_path = Path("../data/statue/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    if len(image_files) >= 2:
        image1 = image_files[0]  # First image
        image2 = image_files[1]  # Second image
        
        # Find matches between the two images
        result = match_features_between_images(image1, image2)
        
        if result:
            matches, kp1, kp2 = result
            print(f"\n✅ Successfully found {len(matches)} mutual matches!")
            
            # Visualize the matches
            matched_image = visualize_matches(image1, image2, matches, kp1, kp2)
            
            # Save visualization
            output_path = Path("../data/statue/feature_matches.jpg")
            cv2.imwrite(str(output_path), matched_image)
            print(f"Saved match visualization to: {output_path}")
        else:
            print("❌ Matching failed")
    else:
        print("Need at least 2 images for matching test")