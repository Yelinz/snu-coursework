import numpy as np

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation


def PnP(X, x):
    """
    Implement the linear perspective-n-point algorithm

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    """
    n_points = x.shape[0]
    A = np.zeros((2 * n_points, 12))
    
    # Build the DLT matrix based on point correspondences
    for i in range(n_points):
        # First row of constraints for x coordinate
        A[2*i, 0:4] = [X[i, 0], X[i, 1], X[i, 2], 1]
        A[2*i, 8:12] = [-x[i, 0] * X[i, 0], -x[i, 0] * X[i, 1], -x[i, 0] * X[i, 2], -x[i, 0]]
        
        # Second row of constraints for y coordinate
        A[2*i+1, 4:8] = [X[i, 0], X[i, 1], X[i, 2], 1]
        A[2*i+1, 8:12] = [-x[i, 1] * X[i, 0], -x[i, 1] * X[i, 1], -x[i, 1] * X[i, 2], -x[i, 1]]

    # Perform SVD to find the solution vector
    _, _, Vh = np.linalg.svd(A)
    P = Vh[-1].reshape((3, 4))

    # Extract and enforce camera matrix properties
    # Decompose rotation matrix using SVD
    U, D, Vh = np.linalg.svd(P[:, :3])
    R = U @ Vh
    t = P[:, 3] / D[0]

    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # Calculate camera center from R and t
    C = -R.T @ t

    return R, C


def PnP_RANSAC(X, x, ransac_n_iter, ransac_thr):
    """
    Estimate pose using PnP with RANSAC

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    inlier : ndarray of shape (n,)
        The indicator of inliers, i.e., the entry is 1 if the point is a inlier,
        and 0 otherwise
    """
    num_points = x.shape[0]
    max_inlier_count = 0

    print(f"Running PnP_RANSAC for {num_points} pairs")
    for _ in range(ransac_n_iter):
        # 1. Randomly sample minimum points needed for pose estimation (6 points)
        sample_indices = np.random.choice(num_points, 6)
        sampled_points_3d = X[sample_indices]
        sampled_points_2d = x[sample_indices]

        # 2. Calculate pose estimate from minimal point set
        rotation_estimate, center_estimate = PnP(sampled_points_3d, sampled_points_2d)

        # 3. Project all 3D points to image using estimated pose
        points_camera_frame = (X - center_estimate) @ rotation_estimate.T
        projected_points_2d = points_camera_frame[:, :2] / points_camera_frame[:, -1:]
        reprojection_error = np.linalg.norm(x - projected_points_2d, axis=1)

        # 4. Calculate reprojection error and count inliers
        inlier_count = np.sum(reprojection_error < ransac_thr)

        # 5. Update best model if iteration has more inliers
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_rotation, best_center = rotation_estimate, center_estimate
            inlier_mask = reprojection_error < ransac_thr

    return best_rotation, best_center, inlier_mask


def ComputePoseJacobian(p, X):
    """
    Compute the pose Jacobian

    Parameters
    ----------
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion
    X : ndarray of shape (3,)
        3D point

    Returns
    -------
    dfdp : ndarray of shape (2, 7)
        The pose Jacobian
    """
    C = p[:3]
    q = p[3:]
    R = Quaternion2Rotation(q)
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = -R[0, :]
    dv_dc = -R[1, :]
    dw_dc = -R[2, :]
    # df_dc is in shape (2, 3)
    df_dc = np.stack(
        [(w * du_dc - u * dw_dc) / (w**2), (w * dv_dc - v * dw_dc) / (w**2)], axis=0
    )

    # du_dR = np.concatenate([X-C, np.zeros(3), X-C])
    # dv_dR = np.concatenate([np.zeros(3), X-C, X-C])
    # dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X-C])
    du_dR = np.concatenate([X - C, np.zeros(3), np.zeros(3)])
    dv_dR = np.concatenate([np.zeros(3), X - C, np.zeros(3)])
    dw_dR = np.concatenate([np.zeros(3), np.zeros(3), X - C])
    # df_dR is in shape (2, 9)
    df_dR = np.stack(
        [(w * du_dR - u * dw_dR) / (w**2), (w * dv_dR - v * dw_dR) / (w**2)], axis=0
    )

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    # dR_dq is in shape (9, 4)
    dR_dq = np.asarray(
        [
            [0, 0, -4 * qy, -4 * qz],
            [-2 * qz, 2 * qy, 2 * qx, -2 * qw],
            [2 * qy, 2 * qz, 2 * qw, 2 * qx],
            [2 * qz, 2 * qy, 2 * qx, 2 * qw],
            [0, -4 * qx, 0, -4 * qz],
            [-2 * qx, -2 * qw, 2 * qz, 2 * qy],
            [-2 * qy, 2 * qz, -2 * qw, 2 * qx],
            [2 * qx, 2 * qw, 2 * qz, 2 * qy],
            [0, -4 * qx, -4 * qy, 0],
        ]
    )

    dfdp = np.hstack([df_dc, df_dR @ dR_dq])

    return dfdp


def PnP_nl(R, C, X, x):
    """
    Update the pose using the pose Jacobian

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix refined by PnP
    c : ndarray of shape (3,)
        Camera center refined by PnP
    X : ndarray of shape (n, 3)
        Set of reconstructed 3D points
    x : ndarray of shape (n, 2)
        2D points of the new image

    Returns
    -------
    R_refined : ndarray of shape (3, 3)
        The rotation matrix refined by nonlinear optimization
    C_refined : ndarray of shape (3,)
        The camera center refined by nonlinear optimization
    """
    n = X.shape[0]
    q = Rotation2Quaternion(R)

    p = np.concatenate([C, q])
    n_iters = 20
    lamb = 1
    error = np.empty((n_iters,))
    for i in range(n_iters):
        R_i = Quaternion2Rotation(p[3:])
        C_i = p[:3]

        proj = (X - C_i[np.newaxis, :]) @ R_i.T
        proj = proj[:, :2] / proj[:, 2, np.newaxis]

        H = np.zeros((7, 7))
        J = np.zeros(7)
        for j in range(n):
            dfdp = ComputePoseJacobian(p, X[j, :])
            H = H + dfdp.T @ dfdp
            J = J + dfdp.T @ (x[j, :] - proj[j, :])

        delta_p = np.linalg.inv(H + lamb * np.eye(7)) @ J
        p += delta_p
        p[3:] /= np.linalg.norm(p[3:])

        error[i] = np.linalg.norm(proj - x)

    R_refined = Quaternion2Rotation(p[3:])
    C_refined = p[:3]
    return R_refined, C_refined
