import cv2
import numpy as np
from pathlib import Path

# Extract ORB features

def extract_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

if __name__ == "__main__":
    print("Feature extraction completed successfully")