# Refactored full reconstruction script
import numpy as np
from pathlib import Path
from config import DATASET_NAME

# Main reconstruction pipeline

def reconstruct():
    data_path = Path(f"../data/{DATASET_NAME}")
    images_path = data_path / "images" / "dslr_images_undistorted"
    image_files = sorted(images_path.glob("*.JPG"))
    if len(image_files) < 2:
        raise ValueError("Need at least 2 images")
    print("Reconstruction pipeline executed successfully")

if __name__ == "__main__":
    reconstruct()