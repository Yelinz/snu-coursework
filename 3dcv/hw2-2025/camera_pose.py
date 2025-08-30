import numpy as np

from feature import EstimateE_RANSAC

def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    # Define the skew-symmetric matrix W for SVD decomposition method
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, Vt = np.linalg.svd(E)

    # Store the four configurations
    R_set = np.zeros((4, 3, 3))
    C_set = np.zeros((4, 3))

    # Loop through the four possible configurations
    config_idx = 0
    for W_matrix in [W, W.T]:  # Two possible rotations: W and W.T
        for sign in [1, -1]:   # Two possible centers: positive and negative
            # Calculate R and C for this configuration
            R = U @ W_matrix @ Vt
            C = sign * U[:, 2]
            
            # Ensure R is a proper rotation matrix (det=1)
            if np.linalg.det(R) < 0:
                R = -R
                C = -C
                
            # Store the results
            R_set[config_idx] = R
            C_set[config_idx] = C
            config_idx += 1

    return R_set, C_set


def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    n = track1.shape[0]
    X = np.zeros((n, 3))

    for i in range(n):
        # Create homogeneous coordinates for 2D points
        u = np.array([track1[i, 0], track1[i, 1], 1])
        v = np.array([track2[i, 0], track2[i, 1], 1])

        # Create cross product matrix [u]× for first image
        skew_u = np.array([[0, -u[2], u[1]], 
                           [u[2], 0, -u[0]], 
                           [-u[1], u[0], 0]])
        
        # Create cross product matrix [v]× for second image
        skew_v = np.array([[0, -v[2], v[1]], 
                           [v[2], 0, -v[0]], 
                           [-v[1], v[0], 0]])
        
        # Take only first two rows from each constraint
        h1 = (skew_u @ P1)[(0, 1), :]
        h2 = (skew_v @ P2)[(0, 1), :]
        
        # Combine constraints to form a system: A·X = 0
        A = np.vstack((h1, h2))
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        homogeneous_X = Vt[-1]
        
        # Convert from homogeneous to Euclidean coordinates
        X[i] = homogeneous_X[:3] / homogeneous_X[3]

    return X


def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    # Extract rotation matrices and translation vectors from projection matrices
    R1, R2 = P1[:, :3], P2[:, :3]
    t1, t2 = P1[:, 3], P2[:, 3]
    # Compute camera centers: C = -R^T * t
    C1, C2 = -R1.T @ t1, -R2.T @ t2
    # Get third row of rotation matrices (z-axis of camera frame)
    r3_1, r3_2 = R1[2, :], R2[2, :]

    # Check if points are in front of camera
    mask1 = ((X - C1) @ r3_1) > 0
    mask2 = ((X - C2) @ r3_2) > 0
    # Points must be in front of both cameras
    valid_index = (mask1 & mask2).reshape(-1)

    return valid_index


def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    max_valid_points = 0
    # Filter valid correspondences
    valid_points_mask = (np.sum(track1, axis=1) != -2) & (np.sum(track2, axis=1) != -2)
    filtered_points1 = track1[valid_points_mask]
    filtered_points2 = track2[valid_points_mask]
    
    # Keep track of original indices for reconstructing the full point array later
    valid_point_indices = np.asarray(np.nonzero(valid_points_mask)[0])

    # Estimate the essential matrix using RANSAC
    E, _ = EstimateE_RANSAC(filtered_points1, filtered_points2, 500, 0.003)

    # Get the four possible camera pose configurations
    R_set, C_set = GetCameraPoseFromE(E)

    # Define the first camera projection matrix (identity matrix)
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # Test all four possible camera pose configurations
    for i in range(4):
        # Create projection matrix for second camera
        P2 = np.hstack([R_set[i], -(R_set[i] @ C_set[i]).reshape((3, 1))])
        
        # Triangulate 3D points using this camera configuration
        triangulated_points = Triangulation(P1, P2, filtered_points1, filtered_points2)
        
        # Check cheirality
        valid_points_mask = EvaluateCheirality(P1, P2, triangulated_points)
        valid_point_count = np.sum(valid_points_mask)
        
        # Update the best solution
        if valid_point_count > max_valid_points:
            max_valid_points = valid_point_count
            best_R = R_set[i]
            best_C = C_set[i]
            
            # Create full-sized array with invalid points set to -1
            best_X = -1 * np.ones((track1.shape[0], 3))
            
            # Copy valid points to their correct positions
            best_X[valid_point_indices[valid_points_mask]] = triangulated_points[valid_points_mask]

    return best_R, best_C, best_X