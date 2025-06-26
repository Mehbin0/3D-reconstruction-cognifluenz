import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.load_images()

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
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask_pose

class Triangulator:
    def __init__(self, K):
        self.K = K

    def triangulate(self, kp1, kp2, matches, R, t):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = np.hstack((R, t))
        P1 = self.K @ proj1
        P2 = self.K @ proj2
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
    fx, fy, cx, cy = 1000, 1000, 640, 360  # Example intrinsics, replace with real values

    # --- Load images ---
    loader = DataLoader(dataset_path)
    if len(loader.images) < 2:
        print("Need at least two images for reconstruction.")
        return

    # --- Feature extraction ---
    extractor = FeatureExtractor()
    kp1, desc1 = extractor.extract(loader.images[0])
    kp2, desc2 = extractor.extract(loader.images[1])

    # --- Feature matching ---
    matcher = FeatureMatcher()
    matches = matcher.match(desc1, desc2)
    print(f"Found {len(matches)} matches.")

    # --- Camera intrinsics ---
    intrinsics = CameraIntrinsics(fx, fy, cx, cy)

    # --- Pose estimation ---
    pose_estimator = PoseEstimator(intrinsics.K)
    R, t, mask_pose = pose_estimator.estimate(kp1, kp2, matches)
    print("Estimated pose between first two images.")

    # --- Triangulation (Sparse reconstruction) ---
    triangulator = Triangulator(intrinsics.K)
    pts3d = triangulator.triangulate(kp1, kp2, matches, R, t)
    print(f"Triangulated {pts3d.shape[0]} 3D points.")

    # --- Visualization (Sparse) ---
    Visualizer.plot_point_cloud(pts3d, title="Sparse 3D Point Cloud")

    # --- Dense reconstruction (optional) ---
    dense_reconstructor = DenseReconstructor(intrinsics.K)
    disparity = dense_reconstructor.compute_depth_map(loader.images[0], loader.images[1])
    points_dense, colors_dense = dense_reconstructor.depth_to_point_cloud(disparity, loader.images[0])
    print(f"Dense point cloud has {points_dense.shape[0]} points.")

    # --- Visualization (Dense) ---
    Visualizer.plot_point_cloud(points_dense, colors_dense, title="Dense 3D Point Cloud")

if __name__ == "__main__":
    main()
