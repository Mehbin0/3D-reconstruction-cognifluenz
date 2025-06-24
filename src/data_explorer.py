import os
from pathlib import Path
from config import DATASET_NAME

def explore_dataset():
    """Our first function - let's see what data we have"""
    
    # Define the path to our statue data
    data_path = Path(f"../data/{DATASET_NAME}")  # Go up one level, then into data/statue
    
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


    # Let's read and parse the cameras.txt file
    print("\n=== Camera Information ===")
    cameras_file = calibration_path / "cameras.txt"
    if cameras_file.exists():
        with open(cameras_file, 'r') as f:
            lines = f.readlines()
        
        # Skip comment lines and find camera data
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                camera_id = parts[0]
                model = parts[1]
                width = parts[2]
                height = parts[3]
                fx = parts[4]  # focal length x
                fy = parts[5]  # focal length y
                cx = parts[6]  # principal point x (center)
                cy = parts[7]  # principal point y (center)
                
                print(f"Camera ID: {camera_id}")
                print(f"Model: {model}")
                print(f"Image size: {width} x {height} pixels")
                print(f"Focal length: fx={fx}, fy={fy}")
                print(f"Principal point: cx={cx}, cy={cy}")
                break  # Just process first camera
    
    # Let's look at camera poses for each image (FIXED for ETH3D format)
    print("\n=== Camera Poses ===")
    images_file = calibration_path / "images.txt"
    if images_file.exists():
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        print("Camera positions for each image:")
        count = 0
        i = 0
        while i < len(lines) and count < 3:
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 10:  # Make sure it's the image info line, not points line
                    image_id = parts[0]
                    image_name = parts[9]
                    tx, ty, tz = parts[5], parts[6], parts[7]  # Camera position
                    
                    print(f"  Image {image_id}: {image_name}")
                    print(f"    Camera position: ({tx}, {ty}, {tz})")
                    count += 1
                    i += 2  # Skip the next line (points data)
                else:
                    i += 1
            else:
                i += 1


if __name__ == "__main__":
    explore_dataset()