import cv2
import numpy as np


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images

    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """
    # Compute pairwise distances
    dists_1to2 = np.linalg.norm(des1[:, None, :] - des2[None, :, :], axis=2)
    dists_2to1 = np.linalg.norm(des2[:, None, :] - des1[None, :, :], axis=2)

    ratio = 0.8
    # Forward match: des1 -> des2
    nn_idx_1to2 = np.argsort(dists_1to2, axis=1)
    matches_1to2 = []
    for i in range(des1.shape[0]):
        idx1 = nn_idx_1to2[i, 0]
        idx2 = nn_idx_1to2[i, 1]
        if dists_1to2[i, idx1] < ratio * dists_1to2[i, idx2]:
            matches_1to2.append((i, idx1))

    # Backward match: des2 -> des1
    nn_idx_2to1 = np.argsort(dists_2to1, axis=1)
    matches_2to1 = []
    for j in range(des2.shape[0]):
        idx1 = nn_idx_2to1[j, 0]
        idx2 = nn_idx_2to1[j, 1]
        if dists_2to1[j, idx1] < ratio * dists_2to1[j, idx2]:
            matches_2to1.append((j, idx1))

    # Bidirectional consistency check
    matches_2to1_dict = {j1: j2 for j1, j2 in matches_2to1}
    matches = []
    for i1, i2 in matches_1to2:
        if matches_2to1_dict.get(i2) == i1:
            matches.append((i1, i2))

    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=int)

    ind1 = np.array([m[0] for m in matches])
    ind2 = np.array([m[1] for m in matches])
    x1 = loc1[ind1]
    x2 = loc2[ind2]

    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    # Normalize points
    def normalize_points(pts):
        mean = np.mean(pts, axis=0)
        std = np.std(pts)
        if std < 1e-8:
            std = 1
        T = np.array(
            [[1 / std, 0, -mean[0] / std], [0, 1 / std, -mean[1] / std], [0, 0, 1]]
        )
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_norm = (T @ pts_h.T).T
        return pts_norm[:, :2], T

    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)

    # Build matrix A
    n = x1.shape[0]
    A = np.zeros((n, 9))
    for i in range(n):
        u1, v1 = x1_norm[i]
        u2, v2 = x2_norm[i]
        A[i] = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1]

    # Solve for E using SVD
    _, _, Vt = np.linalg.svd(A)
    E_norm = Vt[-1].reshape(3, 3)

    # Enforce rank 2 and singular values (1, 1, 0)
    U, S, Vt = np.linalg.svd(E_norm)
    S = np.diag([1, 1, 0])
    E_norm = U @ S @ Vt

    # Denormalize
    E = T2.T @ E_norm @ T1

    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    n = x1.shape[0]
    max_inliers = 0
    inlier_indicies = None
    best_E = None

    # Homogenous coordinates
    x1_h = np.hstack([x1, np.ones((n, 1))])  # (n, 3)
    x2_h = np.hstack([x2, np.ones((n, 1))])  # (n, 3)

    for _ in range(ransac_n_iter):
        # Randomly sample 8 correspondences
        idx = np.random.choice(n, 8, replace=False)
        x1_sample = x1[idx]
        x2_sample = x2[idx]

        # Estimate E from the sample
        E_candidate = EstimateE(x1_sample, x2_sample)

        error = np.abs(np.diag(x1_h @ E_candidate.T @ x2_h.T))

        inliers = np.sum(error < ransac_thr)

        if inliers > max_inliers:
            max_inliers = inliers
            best_E = E_candidate
            inlier_indicies = np.array(np.nonzero(error < ransac_thr)).reshape(-1)

    return best_E, inlier_indicies


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    N = Im.shape[0]
    sift = cv2.xfeatures2d.SIFT_create()
    locs = []
    descs = []

    # 1: for i = 0,··· ,N −1 do
    for i in range(N):
        # 2: Extract SIFT descriptor of the ith image, Im[i]
        gray = cv2.cvtColor(Im[i], cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        loc = np.array([k.pt for k in kp])
        locs.append(loc)
        descs.append(des)
    # 3: end for

    print("BuildFeatureTrack: Extracting SIFT features...")
    feature_tracks = np.empty((N, 0, 2))
    for i in range(N):
        # 5: Initialize track_i = -1^N×F×2
        track_i = -1 * np.ones((N, locs[i].shape[0], 2))

        for j in range(i + 1, N):
            # 7: Match features between the ith and jth images ▷ MatchSIFT
            x1, x2, ind1 = MatchSIFT(locs[i], descs[i], locs[j], descs[j])
            if len(x1) == 0:
                continue

            # 8: Normalize coordinate by multiplying the inverse of intrinsics.
            x1_normalized = np.insert(x1, 2, 1, axis=1) @ np.linalg.inv(K).T
            x2_normalized = np.insert(x2, 2, 1, axis=1) @ np.linalg.inv(K).T
            x1_norm = x1_normalized[:, :2]
            x2_norm = x2_normalized[:, :2]

            # 9: Find inliner matches using essential matrix ▷ EstimateE RANSAC
            _, inliers = EstimateE_RANSAC(
                x1_norm, x2_norm, ransac_n_iter=500, ransac_thr=0.01
            )
            if len(inliers) == 0:
                print("0 RANSAC inliers found for this pair of images.")
                continue

            # 10: Update track_i using the inlier matches.
            track_index = ind1[inliers]

            track_i[i, track_index, :] = x1_norm[inliers]
            track_i[j, track_index, :] = x2_norm[inliers]
        # 11: end for

        # 12: Remove features in track_i that have not been matched for i + 1,··· ,N.
        mask = np.sum(track_i[i], axis=1) != -2
        filtered_track_i = track_i[:, mask, :]

        # 13: track = track ∪ track_i
        feature_tracks = np.concatenate([feature_tracks, filtered_track_i], axis=1)
    # 14: end for

    print(f"BuildFeatureTrack: Created {feature_tracks.shape[1]} feature tracks")
    print(f"Points in image 1: {np.sum(feature_tracks[0, :, 0] != -1)}")
    print(f"Points in image 2: {np.sum(feature_tracks[1, :, 0] != -1)}")
    return feature_tracks
