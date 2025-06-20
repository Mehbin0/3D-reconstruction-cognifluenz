import os
from pathlib import Path

def explore_dataset():
    """Our first function - let's see what data we have"""
    
    # Define the path to our statue data
    data_path = Path("../data/statue")  # Go up one level, then into data/statue
    
    print("=== ETH3D Dataset Explorer ===")
    
    # Your task: Add one line here to print the data_path
    print(f"Looking in: {data_path}")
    # Check if the path exists
    print(f"Path exists: {data_path.exists()}")

    # Let's see what's inside our statue folder
    print("\n=== Folder Contents ===")
    if data_path.exists():
        for item in data_path.iterdir():
            print(f"Found: {item.name} ({'folder' if item.is_dir() else 'file'})")

    # Let's explore the images folder (corrected path)
    print("\n=== Images Folder ===")
    images_path = data_path / "images" / "dslr_images_undistorted"  # Go deeper!
    print(f"Looking in: {images_path}")

    if images_path.exists():
        # Count how many images we have
        image_files = list(images_path.glob("*.JPG"))  # Look for .JPG files
        print(f"Found {len(image_files)} images")
        
        # Show first few image names
        for i, img_file in enumerate(image_files[:3]):  # Show first 3
            print(f"  Image {i+1}: {img_file.name}")
        
        if len(image_files) > 3:
            print(f"  ... and {len(image_files) - 3} more images")
    else:
        print("Images folder not found!")
    
    # Let's explore the calibration folder
    print("\n=== Calibration Folder ===")
    calibration_path = data_path / "calibration"
    if calibration_path.exists():
        print("Found calibration files:")
        for file in calibration_path.iterdir():
            if file.is_file():
                print(f"  {file.name}")


if __name__ == "__main__":
    explore_dataset()