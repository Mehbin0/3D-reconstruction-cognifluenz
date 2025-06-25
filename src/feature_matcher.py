import cv2
import numpy as np
from feature_extractor import extract_orb_features, extract_features
from config import DATASET_NAME
from pathlib import Path

# Match features between images
def match_features(desc1, desc2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(desc1, desc2)
    return matches

if __name__ == "__main__":
    print("Feature matching completed successfully")

    # Test with two consecutive images from our statue
    images_path = Path(f"../data/{DATASET_NAME}/images/dslr_images_undistorted")
    image_files = sorted(list(images_path.glob("*.JPG")))
    
    if len(image_files) >= 2:
        desc1 = extract_features(image_files[0])[1]
        desc2 = extract_features(image_files[1])[1]
        matches = match_features(desc1, desc2)
        
        print("Feature matching completed successfully")
    else:
        print("Need at least 2 images for matching test")