import numpy as np

EXCLUDED_JOINTS = [
    0,  # Pelvis
    3,  # Spine 1
    6,  # Spine 2
    9,  # Spine 3
    10, # Foot
    11, # Foot
    13, # Collar
    14, # Collar
    15, # Head
    22, # Hand
    23, # Hand
]

ORDER = [
    5,
    1,
    3,
    0,
    2,
    4,
    12,
    10,
    8,
    7,
    9,
    11,
    6,
    13
]

def reconstructionError(predicted_joints, gt_joints): 
    filtered_joints = np.delete(predicted_joints, EXCLUDED_JOINTS, axis=0)
    reordered_joints = filtered_joints[ORDER]
    error = np.mean(np.linalg.norm(reordered_joints - gt_joints, axis=1))
    return error

