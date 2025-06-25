import cv2
import numpy as np
from pathlib import Path
from config import DATASET_NAME

# Validate data setup

def validate_data_setup():
    data_path = Path(f"../data/{DATASET_NAME}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    return data_path

# Compute depth map

def compute_depth_map(image1_path, image2_path, camera_matrix, R, t):
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    stereo_sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=11)
    disparity = stereo_sgbm.compute(img1, img2)
    baseline = np.linalg.norm(t)
    focal_length = camera_matrix[0, 0]
    depth_map = (focal_length * baseline) / (disparity + 1e-6)
    return depth_map

# Generate and save 3D point cloud

def generate_point_cloud(depth_map, camera_matrix):
    height, width = depth_map.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    valid_mask = depth_map > 0
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    valid_depths = depth_map[valid_mask]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    X = (valid_x - cx) * valid_depths / fx
    Y = (valid_y - cy) * valid_depths / fy
    Z = valid_depths
    points_3d = np.column_stack((X, Y, Z))
    output_path = Path(f"../data/{DATASET_NAME}/points3D.txt")
    np.savetxt(output_path, points_3d, fmt="%.6f")
    print(f"3D point cloud saved at: {output_path}")
    return points_3d

# Export 3D point cloud to PLY format

def export_point_cloud_to_ply(points_3d, file_path):
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points_3d:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"3D point cloud exported to PLY format at: {file_path}")

# Main function

def main():
    data_path = validate_data_setup()
    images_path = data_path / "images" / "dslr_images_undistorted"
    image_files = sorted(images_path.glob("*.JPG"))
    if len(image_files) < 2:
        raise ValueError("Need at least 2 images")
    camera_matrix = np.eye(3)  # Placeholder for actual camera matrix
    R, t = np.eye(3), np.array([0, 0, 1])  # Placeholder for actual pose
    depth_map = compute_depth_map(image_files[0], image_files[1], camera_matrix, R, t)
    print("Depth map computed successfully")
    # Save the depth map to a file
    output_path = Path(f"../data/{DATASET_NAME}/depth_map.png")
    # Normalize depth map to fit within 8-bit range
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth_map = normalized_depth_map.astype(np.uint8)
    cv2.imwrite(str(output_path), normalized_depth_map)
    print(f"Normalized depth map saved at: {output_path}")
    points_3d = generate_point_cloud(depth_map, camera_matrix)
    ply_output_path = Path(f"../data/{DATASET_NAME}/points3D.ply")
    export_point_cloud_to_ply(points_3d, ply_output_path)

if __name__ == "__main__":
    main()