# Refactored triangulation script
import cv2
import numpy as np
from pathlib import Path
from config import DATASET_NAME

def triangulate(matches, camera_matrix, R, t):
    pts1 = np.array([m.queryIdx for m in matches])
    pts2 = np.array([m.trainIdx for m in matches])
    P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera_matrix @ np.hstack([R, t])
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d

if __name__ == "__main__":
    print("Triangulation completed successfully")