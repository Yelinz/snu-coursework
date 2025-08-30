import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import scipy.interpolate


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
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    # Forward matching: des1 -> des2
    nn1 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    dists1, idxs1 = nn1.kneighbors(des1)
    ratio_mask1 = dists1[:, 0] < 0.75 * dists1[:, 1]

    # Backward matching: des2 -> des1
    nn2 = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des1)
    dists2, idxs2 = nn2.kneighbors(des2)
    ratio_mask2 = dists2[:, 0] < 0.75 * dists2[:, 1]

    # Bi-directional consistency
    matches = []
    for i1, (i2, valid1) in enumerate(zip(idxs1[:, 0], ratio_mask1)):
        if not valid1:
            continue
        # Check if the best match in des2 points back to this descriptor in des1
        if ratio_mask2[i2] and idxs2[i2, 0] == i1:
            matches.append((i1, i2))

    if matches:
        idx1, idx2 = zip(*matches)
        x1 = loc1[list(idx1)]
        x2 = loc2[list(idx2)]
    else:
        x1 = np.zeros((0, 2))
        x2 = np.zeros((0, 2))

    return x1, x2


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    n = x1.shape[0]
    if n < 4:
        return np.eye(3), np.array([], dtype=int)

    def dlt_homography(points_1, points_2):
        N = points_1.shape[0]
        A = []
        for i in range(N):
            x_1, y_1 = points_1[i]
            x_2, y_2 = points_2[i]
            A.append([x_1, y_1, 1, 0, 0, 0, -x_1*x_2, -y_1*x_2, -x_2])
            A.append([0, 0, 0, x_1, y_1, 1, -x_1*y_2, -y_1*y_2, -y_2])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    best_inliers = []
    best_H = np.eye(3)

    for _ in range(ransac_n_iter):
        idx = np.random.choice(n, 4, replace=False)
        points_1 = x1[idx]
        points_2 = x2[idx]
        H = dlt_homography(points_1, points_2)

        # Project x1 using H
        x1_h = np.hstack([x1, np.ones((n, 1))])
        x2_proj = (H @ x1_h.T).T
        x2_proj = x2_proj[:, :2] / x2_proj[:, 2:3]

        # Check error
        errors = np.linalg.norm(x2_proj - x2, axis=1)
        inliers = np.where(errors < ransac_thr)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    # Recompute H with all inliers
    if len(best_inliers) >= 4:
        best_H = dlt_homography(x1[best_inliers], x2[best_inliers])

    return best_H, best_inliers

def EstimateR(H, K):
    """
    Compute the relative rotation matrix

    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    # Remove camera intrinsics
    R = np.linalg.inv(K) @ H @ K

    # Normalize R
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R *= -1

    return R


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface

    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """
    u, v = np.meshgrid(np.arange(Wc), np.arange(Hc))
    u = (u - K[0, 2]) / K[0, 0]
    v = (v - K[1, 2]) / K[1, 1]
    p = np.zeros((Hc, Wc, 3))
    p[:, :, 0] = u
    p[:, :, 1] = v
    p[:, :, 2] = np.ones((Hc, Wc))
    return p


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane

    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    Hc, Wc, _ = p.shape
    # focal length (radius of cylinder)
    f = K[0, 0]
    # Canvas coordinates
    w = np.arange(Wc)
    h = np.arange(Hc)
    ww, hh = np.meshgrid(w, h)
    phi = ww * 2 * np.pi / Wc

    # 3D points on the cylinder surface
    X = f * np.sin(phi)
    Y = hh - Hc//2
    Z = f * np.cos(phi)
    points_3d = np.stack([X, Y, Z], axis=-1)

    # Apply rotation
    points_3d_flat = points_3d.reshape(-1, 3).T 
    points_3d_rot = R @ points_3d_flat         

    # Project to image plane
    points_2d = K @ points_3d_rot
    points_2d = points_2d / points_2d[2, :]

    u = np.zeros((Hc, Wc, 2), dtype=np.float32)
    u[..., 0] = points_2d[0, :].reshape(Hc, Wc)
    u[..., 1] = points_2d[1, :].reshape(Hc, Wc)

    # Validity mask
    mask = (
        (points_3d_rot[2, :].reshape(Hc, Wc) > 0) &
        (u[..., 0] >= 0) & (u[..., 0] < W) &
        (u[..., 1] >= 0) & (u[..., 1] < H)
    ).astype(np.uint8)

    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas

    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    Hc, Wc, _ = u.shape
    canvas_i = np.zeros((Hc, Wc, 3), dtype=np.uint8)

    for channel in range(3):
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (np.arange(image_i.shape[0]), np.arange(image_i.shape[1])),
            image_i[..., channel],
            bounds_error=False,
            fill_value=0
        )
        coords = np.stack([u[..., 1], u[..., 0]], axis=-1)
        interpolated = interpolator(coords.reshape(-1, 2)).reshape(Hc, Wc)
        canvas_i[..., channel] = np.clip(interpolated, 0, 255).astype(np.uint8)

    canvas_i[mask_i == 0] = 0

    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image

    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """

    canvas[mask_i == 1] = canvas_i[mask_i == 1]
    return canvas


if __name__ == "__main__":
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([[320, 0, 480], [0, 320, 270], [0, 0, 1]])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = "{}.jpg".format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        image_1 = im_list[i]
        image_2 = im_list[i + 1]

        # Extract SIFT features
        sift = cv2.SIFT_create()
        keypoints1, des1 = sift.detectAndCompute(image_1, None)
        keypoints2, des2 = sift.detectAndCompute(image_2, None)
        # cv2.imwrite('sift_keypoints.jpg', cv2.drawKeypoints(image_1, loc1, None, cv2.COLOR_BGR2RGB))
        loc1 = np.array([l.pt for l in keypoints1])
        loc2 = np.array([l.pt for l in keypoints2])

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

        # Compute R_new (or R_i+1)
        R_new = np.dot(R, rot_list[i])

        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]

    Hc = Him
    Wc = len(im_list) * Wim // 2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure("HW1")
    plt.axis("off")
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        print(f"outputting image {i + 1}")
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig(
            "output_{}.png".format(i + 1), dpi=600, bbox_inches="tight", pad_inches=0
        )
