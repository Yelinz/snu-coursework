import os
import cv2
import numpy as np

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


if __name__ == "__main__":
    np.random.seed(100)
    K = np.asarray([[463.1, 0, 333.2], [0, 463.1, 187.5], [0, 0, 1]])
    num_images = 14
    w_im = 672
    h_im = 378

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = f"./images/image{i + 1}.jpg"
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i, :, :, :] = im

    # Build feature track
    track = BuildFeatureTrack(Im, K)

    track1 = track[0, :, :]
    track2 = track[1, :, :]
    print("track1 valid points:", np.sum(np.all(track1 != -1, axis=1)))
    print("track2 valid points:", np.sum(np.all(track2 != -1, axis=1)))

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = "output"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))

    # Set first two camera poses
    P[0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P[1] = np.hstack([R, -(R @ C).reshape((3, 1))])

    ransac_n_iter = 1000
    ransac_thr = 0.5
    for i in range(2, num_images):
        X_mask = (X[:, 0] != -1) & (X[:, 1] != -1) & (X[:, 2] != -1)
        track_i = track[i, :, :]
        track_mask_i = (track_i[:, 0] != -1) & (track_i[:, 1] != -1)
        mask = X_mask & track_mask_i

        # Estimate pose using PnP
        R, C, inlier = PnP_RANSAC(X[mask], track_i[mask], ransac_n_iter, ransac_thr)
        R, C = PnP_nl(R, C, X[mask], track_i[mask])

        # Add new camera pose to the set
        P[i] = np.hstack([R, -(R @ C).reshape((3, 1))])

        print(
            f"Successfully estimated pose for image {i + 1} with {np.sum(inlier)} inliers"
        )

        # Now find new points to reconstruct
        candidate_points = FindMissingReconstruction(X, track[i, :, :])
        print(f"Found {np.sum(candidate_points)} new candidate points")

        # Try to triangulate with all previous views
        for j in range(i):
            # Only reconstruct points that are visible in both views
            track_j = track[j, :, :]
            track_mask_j = (track_j[:, 0] != -1) & (track_j[:, 1] != -1)

            mask = (candidate_points.astype(bool)) & track_mask_j
            mask_pos = np.asarray(np.nonzero(mask)[0])

            print(
                f"  Triangulating {np.sum(mask)} points between images {j + 1}-{i + 1}"
            )

            # Triangulate points
            missing_X = Triangulation(P[i], P[j], track_i[mask], track_j[mask])
            missing_X = Triangulation_nl(
                missing_X, P[i], P[j], track_i[mask], track_j[mask]
            )

            # Filter out points based on cheirality
            valid_points = EvaluateCheirality(P[i], P[j], missing_X)

            # Update 3D points
            X[mask_pos[valid_points]] = missing_X[valid_points]

        # Run bundle adjustment with all reconstructed points
        valid_poses = X[:, 0] != -1
        X_ba = X[valid_poses, :]
        track_current = track[: i + 1, valid_poses, :]
        P_latest, X_new = RunBundleAdjustment(P[: i + 1, :, :], X_ba, track_current)
        P[: i + 1, :, :] = P_latest
        X[valid_poses, :] = X_new

        ###############################################################
        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i + 1):
            R_d = P[j, :, :3]
            C_d = -R_d.T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh("{}/cameras_{}.ply".format(output_dir, i + 1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0], 1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j, :, :].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (
                (x[:, 0] >= -1) * (x[:, 0] <= 1) * (x[:, 1] >= -1) * (x[:, 1] <= 1)
            )
            uv = x[mask_valid, :] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(
                    np.arange(h_im),
                    np.arange(w_im),
                    Im[j, :, :, k].astype(float) / 255,
                    kx=1,
                    ky=1,
                )
                colors[mask_valid, k] = interp_fun(uv[:, 1], uv[:, 0], grid=False)

        ind = np.sqrt(np.sum(X_ba**2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud("{}/points_{}.ply".format(output_dir, i + 1), pcd)
