import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.load_images() # Load images from the dataset path as we initialize the DataLoader

    def load_images(self):
        image_dir = os.path.join(self.dataset_path, "images") 
        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_COLOR)
                if img is not None:
                    self.images.append(img)
        print(f"Loaded {len(self.images)} images.")

class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(5000)

    def extract(self, image):
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

class FeatureMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, desc1, desc2):
        matches = self.matcher.match(desc1, desc2)
        # matches is a list of cv2.DMatch objects
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

class CameraIntrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)

class PoseEstimator:
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2

    def estimate(self, kp1, kp2, matches):
        # Get numpy array of keypoints' coordinates in the matches
        # m.queryIdx is the index of the keypoint in the first image
        # kp1[m.queryIdx].pt gives the (x, y) coordinates of that keypoint
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        # ptsx is the numpy array of points in the first image which are matched to ptsy
        # Normalize points using each camera's intrinsics
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K1, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K2, None).reshape(-1, 2)
        # Estimate essential matrix using normalized points
<<<<<<< HEAD
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, self.K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # RANSAC creates models trained on subsets of the matches, and selects the model with maximum possible error 1.0 and confidence 0.999
        # mask is an array of 0s and 1s, where 1 means the match is inliers and 0 means outliers
        # Recover pose using normalized points
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_norm, pts2_norm, self.K1)
=======
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K1, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # RANSAC creates models trained on subsets of the matches, and selects the model with maximum possible error 1.0 and confidence 0.999
        # mask is an array of 0s and 1s, where 1 means the match is inliers and 0 means outliers
        # Recover pose using normalized points
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K1)
>>>>>>> 8a5dd391f68343b3c8b5fabc82862f91cef0634e

        return R, t, mask_pose
    
class Triangulator:
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2

    def triangulate(self, kp1, kp2, matches, R, t):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # proj1 = [[1, 0, 0, 0],
        #          [0, 1, 0, 0],            
        #          [0, 0, 1, 0]]
        proj2 = np.hstack((R, t))
        # proj2 defines the second camera's pose in the world coordinate system with respect to the first camera
        # self.K1 is the intrinsic matrix of the camera 1 (focal lengths and principal point)
        P1 = self.K1 @ proj1
        P2 = self.K2 @ proj2
        # @ is the matrix multiplication operator in Python
        # P1 and P2 are the projection matrices for the first and second cameras respectively
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        return pts3d.T

class DenseReconstructor:
    def __init__(self, K1, K2, baseline):
        self.K1 = K1
        self.K2 = K2
        self.baseline = baseline  # Baseline distance between the two cameras

    def compute_depth_map(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        grayL = grayL.astype(np.uint8)
        grayR = grayR.astype(np.uint8)
        if grayL.shape != grayR.shape:
            print(f"Warning: Image sizes do not match! Left: {grayL.shape}, Right: {grayR.shape}")
            # Optionally, resize right to match left:
            print("Resizing right image to match left image size.")
            grayR = cv2.resize(grayR, (grayL.shape[1], grayL.shape[0]))
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            # size of block to match
            blockSize=9,
            P1=8 * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        return disparity

    def depth_to_point_cloud(self, disparity, imgL):
        h, w = disparity.shape
        # Use the left camera's intrinsics for Q matrix
        fx = self.K1[0, 0]
        cx = self.K1[0, 2]
        cy = self.K1[1, 2]
        # Estimate baseline from translation vector (assume t is in meters)
        # If you have the actual baseline, use it here
        baseline = self.baseline
        Q = np.float32([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1 / baseline, 0]
        ])
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        mask = disparity > disparity.min()
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        out_points = points_3d[mask]
        out_colors = colors[mask]
        return out_points, out_colors

class Visualizer:
    # staticmethod allows us to call this method without creating an instance of Visualizer
    @staticmethod
    def plot_point_cloud(points, colors=None, title="3D Point Cloud", use_open3d=False):
        if use_open3d:
            # Use Open3D for interactive visualization
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                # Ensure colors are in [0, 1] range
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd], window_name=title)
        else:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            if colors is not None:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors/255.0, s=1)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)
            ax.set_title(title)
            plt.show()

def main():
    # --- User parameters ---
    # List available datasets in the 'data' folder
    data_root = "data"
    datasets = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    if not datasets:
        print("No datasets found in the 'data' folder.")
        sys.exit(1)
    print("Available datasets:")
    for idx, name in enumerate(datasets):
        print(f"{idx}: {name}")
    ds_idx = input("Select dataset index: ").strip()
    if not ds_idx.isdigit() or int(ds_idx) < 0 or int(ds_idx) >= len(datasets):
        print("Invalid selection.")
        sys.exit(1)
    dataset_name = datasets[int(ds_idx)]
    dataset_path = os.path.join(data_root, dataset_name)

    # Try to read intrinsics from calibration/cameras.txt
    calib_file = os.path.join(dataset_path, "calibration", "cameras.txt")
    fx = fy = cx = cy = None
    camera_intrinsics = {}
    if os.path.isfile(calib_file):
        print(f"Reading intrinsics from {calib_file}...")
        with open(calib_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    # Format: camera_id, model, width, height, fx, fy, cx, cy, ...
                    cam_id = int(parts[0])
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    camera_intrinsics[cam_id] = (fx, fy, cx, cy)
        if len(camera_intrinsics) > 0:
            print(f"Found {len(camera_intrinsics)} cameras in {calib_file}.")
        else:
            print("No valid camera intrinsics found in the calibration file.")
            sys.exit(1)

    image_camera_id = {}
    images_txt = os.path.join(dataset_path, "calibration", "images.txt")
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 10:
                image_name = parts[-1]
                cam_id = int(parts[8])
                image_camera_id[image_name.split("/")[-1]] = cam_id
        
    # --- Load images ---
    loader = DataLoader(dataset_path)
    if len(loader.images) < 2:
        print("Need at least two images for reconstruction.")
        return
    # Images extracted to loader.images

    for idx, fname in enumerate(sorted(os.listdir(os.path.join(dataset_path, 'images')))):
        print(f"{idx}: {fname}")

    extractor = FeatureExtractor()
    matcher = FeatureMatcher()
    all_pts3d = []
    rnt = []
    all_colors = []
    for i in range(len(loader.images) - 1):
        print(f"\nProcessing pair {i} and {i+1}...")
        # Feature extraction
        kp1, desc1 = extractor.extract(loader.images[i])
        kp2, desc2 = extractor.extract(loader.images[i + 1])
        # Feature matching
        matches = matcher.match(desc1, desc2)
        print(f"Found {len(matches)} matches between image {i} and image {i + 1}.") 
        if len(matches) < 8:
            print("Not enough matches to estimate pose. Skipping this pair.")
            continue
        colors = []
        # Collect colors for the matched keypoints in the first image
        for m in matches:
            x, y = map(int, kp1[m.queryIdx].pt)
            colors.append(loader.images[i][y, x])
        # all_colors.append(np.array(colors))
        # Camera intrinsics for each image
        print("Fetching camera intrinsics...")
        image_files = sorted(os.listdir(os.path.join(dataset_path, 'images')))
        fname1 = image_files[i]
        fname2 = image_files[i + 1]
        cam_id1 = image_camera_id[fname1]
        cam_id2 = image_camera_id[fname2]
        intrinsics1 = CameraIntrinsics(*camera_intrinsics[cam_id1])
        intrinsics2 = CameraIntrinsics(*camera_intrinsics[cam_id2])
        print(f"Camera intrinsics for {fname1}: {intrinsics1.K}")
        print(f"Camera intrinsics for {fname2}: {intrinsics2.K}")
        # Pose estimation
        pose_estimator = PoseEstimator(intrinsics1.K, intrinsics2.K)
        R, t, mask_pose = pose_estimator.estimate(kp1, kp2, matches)
        # Append R and t to the list of all poses
        rnt.append((R, t))
        print("Estimated pose between images.")
        # Triangulation
        triangulator = Triangulator(intrinsics1.K, intrinsics2.K)
        pts3d = triangulator.triangulate(kp1, kp2, matches, R, t)
        print(f"Triangulated {pts3d.shape[0]} 3D points.")
        # Convert to world coordinates by reprojecting points using rnt list
        # Compose all previous rotations and translations to get the world pose of the first camera in this pair
        R_world = np.eye(3)
        t_world = np.zeros((3, 1))
        for j in range(i):
            R_prev, t_prev = rnt[j]
            t_world = R_prev @ t_world + t_prev
            R_world = R_prev @ R_world
        # Transform points: X_world = R_world @ X_cam.T + t_world, then transpose back
        pts3d_world = (R_world @ pts3d.T).T + t_world.reshape(1, 3)
        pts3d = pts3d_world
        all_pts3d.append(pts3d)
        if i == 0:
            colors = np.tile(np.array([255, 0, 0]), (pts3d.shape[0], 1))  # Red
        elif i == 1:
            colors = np.tile(np.array([255, 255, 0]), (pts3d.shape[0], 1))  # Yellow
        elif i == 2:
            colors = np.tile(np.array([0, 0, 255]), (pts3d.shape[0], 1))  # Blue
        else:
            colors = np.array(colors)
        all_colors.append(colors)
        # Visualizer.plot_point_cloud(np.vstack(all_pts3d), colors=np.vstack(all_colors), title=f"Sparse 3D Point Cloud from Pair {i} and {i+1}", use_open3d=True)
    # Merge all points into one big array
    if all_pts3d:
        merged_pts3d = np.vstack(all_pts3d)
        merged_colors = np.vstack(all_colors)
        # Remove near-duplicate points (simple approach)
        rounded_pts = np.round(merged_pts3d, decimals=3)
        unique_pts, unique_indices = np.unique(rounded_pts, axis=0, return_index=True)
        # select unique colors based on unique points
        unique_colors = merged_colors[unique_indices]
        print(f"Total unique 3D points from all pairs: {unique_pts.shape[0]}")
        Visualizer.plot_point_cloud(unique_pts, colors=unique_colors, title="Sparse 3D Point Cloud from All Pairs", use_open3d=True)
    else:
        print("No valid 3D points found from the image pairs.")

<<<<<<< HEAD

    # i1 = int(input("Enter index of first image: "))
    # i2 = int(input("Enter index of second image: "))

    # # --- Feature extraction ---
    # # Get keypoints and descriptors for the first two images
    # kp1, desc1 = extractor.extract(loader.images[i1])
    # kp2, desc2 = extractor.extract(loader.images[i2])

    # # --- Feature matching ---
    # matcher = FeatureMatcher()
    # matches = matcher.match(desc1, desc2)
    # # Gives a sorted list of cv2.DMatch objects which are matches between the two images
    # print(f"Found {len(matches)} matches.")

#     # --- Camera intrinsics ---
#     print("Fetching camera intrinsics...")
#     image_files = sorted(os.listdir(os.path.join(dataset_path, 'images')))
#     fname1 = image_files[i1]
#     fname2 = image_files[i2]
#     intrinsics1 = CameraIntrinsics(*camera_intrinsics[image_camera_id[fname1]])
#     print(f"Camera intrinsics for {fname1}: {intrinsics1.K}")
#     intrinsics2 = CameraIntrinsics(*camera_intrinsics[image_camera_id[fname2]])
#     print(f"Camera intrinsics for {fname2}: {intrinsics2.K}")

#     # --- Pose estimation ---
#     pose_estimator = PoseEstimator(intrinsics1.K, intrinsics2.K)
#     R, t, mask_pose = pose_estimator.estimate(kp1, kp2, matches)
#     print("Estimated pose between first two images.")

#     while True:
#         print("\n=== Main Menu ===")
#         print("1. Sparse Reconstruction")
#         print("2. Dense Reconstruction")
#         print("e. Exit")
#         choice = input("Select reconstruction type: ").strip().lower()
#         if choice == 'e':
#             print("Exiting.")
#             sys.exit(0)
#         if choice == '1':
#             triangulator = Triangulator(intrinsics1.K, intrinsics2.K)
#             pts3d = triangulator.triangulate(kp1, kp2, matches, R, t)
#             print(f"Triangulated {pts3d.shape[0]} 3D points.")
#             while True:
#                 print("\n--- Sparse Reconstruction ---")
#                 print("1. Visualize (with optional subsample)")
#                 print("b. Back")
#                 print("e. Exit")
#                 sub_choice = input("Choose an option: ").strip().lower()
#                 if sub_choice == 'e':
#                     print("Exiting.")
#                     sys.exit(0)
#                 if sub_choice == 'b':
#                     break
#                 elif sub_choice == '1':
#                     print(f"Total points: {pts3d.shape[0]}")
#                     subsample = input("Enter subsample size (number of points to visualize, or press Enter to show all): ").strip()
#                     if subsample == "":
#                         subsample_points = pts3d
#                     elif subsample.isdigit() and 0 < int(subsample) <= pts3d.shape[0]:
#                         idx = np.random.choice(pts3d.shape[0], int(subsample), replace=False)
#                         subsample_points = pts3d[idx]
#                     else:
#                         print("Invalid input. Showing all points.")
#                         subsample_points = pts3d
#                     Visualizer.plot_point_cloud(subsample_points, title="Sparse 3D Point Cloud")
#         elif choice == '2':
#             baseline = np.linalg.norm(t)  # Assuming t is the translation vector between the two cameras
#             dense_reconstructor = DenseReconstructor(intrinsics1.K, intrinsics2.K,baseline)
#             disparity = dense_reconstructor.compute_depth_map(loader.images[i1], loader.images[i2])
#             points_dense, colors_dense = dense_reconstructor.depth_to_point_cloud(disparity, loader.images[i1])
#             print(f"Dense point cloud has {points_dense.shape[0]} points.")
#             while True:
#                 print("\n--- Dense Reconstruction ---")
#                 print("1. Visualize (with optional subsample)")
#                 print("2. Save depth map as image")
#                 print("3. Save disparity map as image")
#                 print("b. Back")
#                 print("e. Exit")
#                 sub_choice = input("Choose an option: ").strip().lower()
#                 if sub_choice == 'e':
#                     print("Exiting.")
#                     sys.exit(0)
#                 if sub_choice == 'b':
#                     break
#                 elif sub_choice == '1':
#                     print(f"Total points: {points_dense.shape[0]}")
#                     subsample = input("Enter subsample size (number of points to visualize, or press Enter to show all): ").strip()
#                     if subsample == "":
#                         subsample_points = points_dense
#                         subsample_colors = colors_dense
#                     elif subsample.isdigit() and 0 < int(subsample) <= points_dense.shape[0]:
#                         idx = np.random.choice(points_dense.shape[0], int(subsample), replace=False)
#                         subsample_points = points_dense[idx]
#                         subsample_colors = colors_dense[idx]
#                     else:
#                         print("Invalid input. Showing all points.")
#                         subsample_points = points_dense
#                         subsample_colors = colors_dense
#                     Visualizer.plot_point_cloud(subsample_points, subsample_colors, title="Dense 3D Point Cloud")
#                 elif sub_choice == '2':
#                     # Save depth map as image (normalize for visualization)
#                     depth = points_dense[:, 2]
#                     depth_img = np.zeros_like(disparity, dtype=np.float32)
#                     mask = disparity > disparity.min()
#                     depth_img[mask] = depth
#                     # Remove invalid values before percentile calculation
#                     valid_depths = depth_img[mask]
#                     valid_depths = valid_depths[np.isfinite(valid_depths)]
#                     if valid_depths.size > 0:
#                         # Use robust percentiles to avoid outliers
#                         max_depth = np.percentile(valid_depths, 99)
#                         min_depth = np.percentile(valid_depths, 1)
#                         # Avoid degenerate normalization
#                         if max_depth > min_depth:
#                             clipped = np.clip(depth_img, min_depth, max_depth)
#                             norm_depth = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
#                         else:
#                             norm_depth = np.zeros_like(depth_img, dtype=np.uint8)
#                         norm_depth = np.nan_to_num(norm_depth, nan=0, posinf=0, neginf=0)
#                         norm_depth = norm_depth.astype(np.uint8)
#                     else:
#                         norm_depth = np.zeros_like(depth_img, dtype=np.uint8)
#                     cv2.imwrite("dense_depth_map.png", norm_depth)
#                     print("Saved: dense_depth_map.png")
#                 elif sub_choice == '3':
#                     # Save disparity map as image (normalize for visualization)
#                     disp = disparity.copy()
#                     disp[~np.isfinite(disp)] = 0
#                     norm_disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
#                     norm_disp = norm_disp.astype(np.uint8)
#                     cv2.imwrite("disparity_map.png", norm_disp)
#                     print("Saved: disparity_map.png")
=======
    # --- Camera intrinsics ---
    print("Fetching camera intrinsics...")
    image_files = sorted(os.listdir(os.path.join(dataset_path, 'images')))
    fname1 = image_files[i1]
    fname2 = image_files[i2]
    intrinsics1 = CameraIntrinsics(*camera_intrinsics[image_camera_id[fname1]])
    print(f"Camera intrinsics for {fname1}: {intrinsics1.K}")
    intrinsics2 = CameraIntrinsics(*camera_intrinsics[image_camera_id[fname2]])
    print(f"Camera intrinsics for {fname2}: {intrinsics2.K}")

    # --- Pose estimation ---
    pose_estimator = PoseEstimator(intrinsics1.K, intrinsics2.K)
    R, t, mask_pose = pose_estimator.estimate(kp1, kp2, matches)
    print("Estimated pose between first two images.")

    while True:
        print("\n=== Main Menu ===")
        print("1. Sparse Reconstruction")
        print("2. Dense Reconstruction")
        print("e. Exit")
        choice = input("Select reconstruction type: ").strip().lower()
        if choice == 'e':
            print("Exiting.")
            sys.exit(0)
        if choice == '1':
            triangulator = Triangulator(intrinsics1.K, intrinsics2.K)
            pts3d = triangulator.triangulate(kp1, kp2, matches, R, t)
            print(f"Triangulated {pts3d.shape[0]} 3D points.")
            while True:
                print("\n--- Sparse Reconstruction ---")
                print("1. Visualize (with optional subsample)")
                print("b. Back")
                print("e. Exit")
                sub_choice = input("Choose an option: ").strip().lower()
                if sub_choice == 'e':
                    print("Exiting.")
                    sys.exit(0)
                if sub_choice == 'b':
                    break
                elif sub_choice == '1':
                    print(f"Total points: {pts3d.shape[0]}")
                    subsample = input("Enter subsample size (number of points to visualize, or press Enter to show all): ").strip()
                    if subsample == "":
                        subsample_points = pts3d
                    elif subsample.isdigit() and 0 < int(subsample) <= pts3d.shape[0]:
                        idx = np.random.choice(pts3d.shape[0], int(subsample), replace=False)
                        subsample_points = pts3d[idx]
                    else:
                        print("Invalid input. Showing all points.")
                        subsample_points = pts3d
                    Visualizer.plot_point_cloud(subsample_points, title="Sparse 3D Point Cloud")
        elif choice == '2':
            baseline = np.linalg.norm(t)  # Assuming t is the translation vector between the two cameras
            dense_reconstructor = DenseReconstructor(intrinsics1.K, intrinsics2.K,baseline)
            disparity = dense_reconstructor.compute_depth_map(loader.images[i1], loader.images[i2])
            points_dense, colors_dense = dense_reconstructor.depth_to_point_cloud(disparity, loader.images[i1])
            print(f"Dense point cloud has {points_dense.shape[0]} points.")
            while True:
                print("\n--- Dense Reconstruction ---")
                print("1. Visualize (with optional subsample)")
                print("2. Save depth map as image")
                print("3. Save disparity map as image")
                print("b. Back")
                print("e. Exit")
                sub_choice = input("Choose an option: ").strip().lower()
                if sub_choice == 'e':
                    print("Exiting.")
                    sys.exit(0)
                if sub_choice == 'b':
                    break
                elif sub_choice == '1':
                    print(f"Total points: {points_dense.shape[0]}")
                    subsample = input("Enter subsample size (number of points to visualize, or press Enter to show all): ").strip()
                    if subsample == "":
                        subsample_points = points_dense
                        subsample_colors = colors_dense
                    elif subsample.isdigit() and 0 < int(subsample) <= points_dense.shape[0]:
                        idx = np.random.choice(points_dense.shape[0], int(subsample), replace=False)
                        subsample_points = points_dense[idx]
                        subsample_colors = colors_dense[idx]
                    else:
                        print("Invalid input. Showing all points.")
                        subsample_points = points_dense
                        subsample_colors = colors_dense
                    Visualizer.plot_point_cloud(subsample_points, subsample_colors, title="Dense 3D Point Cloud")
                elif sub_choice == '2':
                    # Save depth map as image (normalize for visualization)
                    depth = points_dense[:, 2]
                    depth_img = np.zeros_like(disparity, dtype=np.float32)
                    mask = disparity > disparity.min()
                    depth_img[mask] = depth
                    # Remove invalid values before percentile calculation
                    valid_depths = depth_img[mask]
                    valid_depths = valid_depths[np.isfinite(valid_depths)]
                    if valid_depths.size > 0:
                        # Use robust percentiles to avoid outliers
                        max_depth = np.percentile(valid_depths, 99)
                        min_depth = np.percentile(valid_depths, 1)
                        # Avoid degenerate normalization
                        if max_depth > min_depth:
                            clipped = np.clip(depth_img, min_depth, max_depth)
                            norm_depth = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
                        else:
                            norm_depth = np.zeros_like(depth_img, dtype=np.uint8)
                        norm_depth = np.nan_to_num(norm_depth, nan=0, posinf=0, neginf=0)
                        norm_depth = norm_depth.astype(np.uint8)
                    else:
                        norm_depth = np.zeros_like(depth_img, dtype=np.uint8)
                    cv2.imwrite("dense_depth_map.png", norm_depth)
                    print("Saved: dense_depth_map.png")
                elif sub_choice == '3':
                    # Save disparity map as image (normalize for visualization)
                    disp = disparity.copy()
                    disp[~np.isfinite(disp)] = 0
                    norm_disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
                    norm_disp = norm_disp.astype(np.uint8)
                    cv2.imwrite("disparity_map.png", norm_disp)
                    print("Saved: disparity_map.png")
>>>>>>> 8a5dd391f68343b3c8b5fabc82862f91cef0634e

if __name__ == "__main__":
    main()
