import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    def __init__(self, K):
        self.K = K

    def estimate(self, kp1, kp2, matches):
        # Get numpy array of keypoints' coordinates in the matches
        # m.queryIdx is the index of the keypoint in the first image
        # kp1[m.queryIdx].pt gives the (x, y) coordinates of that keypoint
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        # ptsx is the numpy array of points in the first image which are matched to ptsy
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # RANSAC creates models trained on subsets of the matches, and selects the model with maximum possible error 1.0 and confidence 0.999
        # mask is an array of 0s and 1s, where 1 means the match is inliers and 0 means outliers
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        # mask_pose is an array of 0s and 1s, where 1 means the match fits  the estimated pose and 0 means it does not
        return R, t, mask_pose
        # More 1s in mask_pose usually means a more reliable pose estimate

class Triangulator:
    def __init__(self, K):
        self.K = K

    def triangulate(self, kp1, kp2, matches, R, t):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # proj1 = [[1, 0, 0, 0],
        #          [0, 1, 0, 0],            
        #          [0, 0, 1, 0]]
        proj2 = np.hstack((R, t))
        # proj2 defines the second camera's pose in the world coordinate system with respect to the first camera
        # self.K is the intrinsic matrix of the camera (focal lengths and principal point)
        P1 = self.K @ proj1
        P2 = self.K @ proj2
        # @ is the matrix multiplication operator in Python
        # P1 and P2 are the projection matrices for the first and second cameras respectively
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        return pts3d.T

class DenseReconstructor:
    def __init__(self, K):
        self.K = K

    def compute_depth_map(self, imgL, imgR):
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
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
        f = self.K[0, 0]
        Q = np.float32([[1, 0, 0, -self.K[0, 2]],
                        [0, -1, 0, self.K[1, 2]],
                        [0, 0, 0, -f],
                        [0, 0, 1, 0]])
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        mask = disparity > disparity.min()
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        out_points = points_3d[mask]
        out_colors = colors[mask]
        return out_points, out_colors

class Visualizer:
    # staticmethod allows us to call this method without creating an instance of Visualizer
    @staticmethod
    def plot_point_cloud(points, colors=None, title="3D Point Cloud"):
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
    dataset_path = "data/statue"  # Change as needed
    fx, fy, cx, cy = 3410.99, 3412.37, 3129.34, 2059.72  # Example intrinsics, replace with real values

    # --- Load images ---
    loader = DataLoader(dataset_path)
    if len(loader.images) < 2:
        print("Need at least two images for reconstruction.")
        return
    # Images extracted to loader.images

    # --- Feature extraction ---
    extractor = FeatureExtractor()
    # Get keypoints and descriptors for the first two images
    kp1, desc1 = extractor.extract(loader.images[0])
    kp2, desc2 = extractor.extract(loader.images[1])

    # --- Feature matching ---
    matcher = FeatureMatcher()
    matches = matcher.match(desc1, desc2)
    # Gives a sorted list of cv2.DMatch objects which are matches between the two images
    print(f"Found {len(matches)} matches.")

    # --- Camera intrinsics ---
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    # intrinsics.K is the camera intrinsic matrix (Calibration matrix)

    # --- Pose estimation ---
    pose_estimator = PoseEstimator(intrinsics.K)
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
            triangulator = Triangulator(intrinsics.K)
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
            dense_reconstructor = DenseReconstructor(intrinsics.K)
            disparity = dense_reconstructor.compute_depth_map(loader.images[0], loader.images[1])
            points_dense, colors_dense = dense_reconstructor.depth_to_point_cloud(disparity, loader.images[0])
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
                    # Normalize to 0-255 for saving as 8-bit image
                    valid_depths = depth_img[mask]
                    valid_depths = valid_depths[np.isfinite(valid_depths)]
                    if valid_depths.size > 0:
                        max_depth = np.percentile(valid_depths, 99)  # or set a fixed value
                        min_depth = np.percentile(valid_depths, 1)
                        clipped = np.clip(depth_img, min_depth, max_depth)
                        norm_depth = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
                        norm_depth = np.nan_to_num(norm_depth, nan=0, posinf=0, neginf=0)
                        norm_depth = norm_depth.astype(np.uint8)
                    else:
                        norm_depth = np.zeros_like(depth_img, dtype=np.uint8)
                    cv2.imwrite("dense_depth_map.png", norm_depth)
                    print("Saved: dense_depth_map.png")
                    # norm_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                    # norm_depth = np.nan_to_num(norm_depth, nan=0, posinf=0, neginf=0)
                    # norm_depth = norm_depth.astype(np.uint8)
                    # cv2.imwrite("dense_depth_map.png", norm_depth)
                    # print("Saved: dense_depth_map.png")
                elif sub_choice == '3':
                    # Save disparity map as image (normalize for visualization)
                    disp = disparity.copy()
                    disp[disp == np.inf] = 0
                    disp[disp == -np.inf] = 0
                    disp = np.nan_to_num(disp)
                    norm_disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
                    norm_disp = norm_disp.astype(np.uint8)
                    cv2.imwrite("disparity_map.png", norm_disp)
                    print("Saved: disparity_map.png")
                    
    # # --- Triangulation (Sparse reconstruction) ---
    # triangulator = Triangulator(intrinsics.K)
    # pts3d = triangulator.triangulate(kp1, kp2, matches, R, t)
    # print(f"Triangulated {pts3d.shape[0]} 3D points.")

    # # --- Visualization (Sparse) ---
    # Visualizer.plot_point_cloud(pts3d, title="Sparse 3D Point Cloud")

    # # --- Dense reconstruction (optional) ---
    # dense_reconstructor = DenseReconstructor(intrinsics.K)
    # disparity = dense_reconstructor.compute_depth_map(loader.images[0], loader.images[1])
    # points_dense, colors_dense = dense_reconstructor.depth_to_point_cloud(disparity, loader.images[0])
    # print(f"Dense point cloud has {points_dense.shape[0]} points.")

    # # --- Visualization (Dense) ---
    # Visualizer.plot_point_cloud(points_dense, colors_dense, title="Dense 3D Point Cloud")

if __name__ == "__main__":
    main()
