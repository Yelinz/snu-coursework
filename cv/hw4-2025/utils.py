import numpy as np
import json
import pickle
import trimesh

from pathlib import Path
from trimesh.transformations import translation_matrix


N_JNTS = 19  # Number of joints in the body keypoints


def load_cam_data(cam_pth: Path, intr_pth: Path):
    """
    return:
    Dict[
        (int) cam_id: Dict[
            c2w: np.array(shape=(4, 4)),
            height: int,
            width: int,
            intrinsics: np.array(shape=(3, 3))
        ]
    ]
    """
    cam_data = np.load(cam_pth)
    new_cam_dict = dict()
    for k, c2w in cam_data.items():
        cam_id = int(k)

        c2w = c2w.astype(np.float32)
        new_cam_dict[cam_id] = dict(
            c2w=c2w,
        )
    
    with open(intr_pth, 'r') as f:
        intrinsic_data = json.load(f)
        
        
    for k, v in intrinsic_data.items():
        cam_id = int(k)
        if cam_id not in new_cam_dict:
            print(f"Camera ID {cam_id} not found in camera data.")
            continue
        new_cam_dict[cam_id]['height'] = v['height']
        new_cam_dict[cam_id]['width'] = v['width']
        new_cam_dict[cam_id]['intrinsics'] = np.array(v['Intrinsics'], dtype=np.float32).reshape(3,3)   # we need undistorted intrinsics only
    
    return new_cam_dict
    
    
def load_body_kpts(pose_dir: Path):
    """
    Load pose data from a directory containing .pkl files.
    Returns a dictionary mapping camera IDs to pose data.
    """
    body_kpts_dict = dict()
    for pose_fname in pose_dir.glob("*.pkl"):
        cam_id = int(pose_fname.stem.split('.')[0])
        with open(pose_fname, 'rb') as f:
            pose = pickle.load(f)  # Assuming the pose is stored in JSON format

        # load pose data
        if 'body_keypoints' not in pose:
            print(f"Warning: 'body_keypoints' not found in pose data for camera {cam_id}.")
            continue

        kpts = []
        for pid in sorted(list(pose['body_keypoints'])):
            kpt = np.array(pose['body_keypoints'][pid], dtype=np.float32)
            kpts.append(kpt)
        
        body_kpts_dict[cam_id] = np.stack(kpts, dtype=np.float32, axis=0) # (N, J, 3)
    return body_kpts_dict



def get_bones():
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], \
               [10, 11], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [19, 9], [19, 12], [2, 19], \
               [1, 16], [16, 18]]  # , [3, 17], [6, 18]]
    
    # [2, 9], [2, 12], 
    limbSeq = [[int(u)-1, int(v)-1] for u, v in limbSeq]

    return limbSeq






def create_bones(part_scores, joint_pairs, node_proposals, viz_thrs=0.05):

    part_meshes = dict()
    n_parts = 0
    for part_idx, (u, v) in enumerate(joint_pairs):
        u_nodes = node_proposals[u]
        v_nodes = node_proposals[v]
        part_meshes[part_idx] = []

        scores = part_scores[part_idx]

        if scores is None:
            continue

        for i in range(len(scores)):
            for j in range(len(scores[i])):
                if scores[i][j] > viz_thrs:
                    A = u_nodes[i]
                    B = v_nodes[j]
                    bone_mesh = create_bone_mesh(A, B)
                    
                    part_meshes[part_idx].append(dict(
                        verts=bone_mesh.vertices,
                        faces=bone_mesh.faces,
                        score= scores[i][j],
                    ))
                    n_parts += 1

    print(f"Total {n_parts} parts for viz created.")

    return part_meshes



def create_bone_mesh(A, B, radius=0.01, sections=32):
    """
    Create a trimesh cylinder mesh connecting points A and B.

    Parameters:
        A: (3,) array_like, start point
        B: (3,) array_like, end point
        radius: float, radius of the cylinder
        sections: int, number of radial segments

    Returns:
        trimesh.Trimesh object
    """
    A = np.array(A)
    B = np.array(B)
    vec = B - A
    height = np.linalg.norm(vec)

    # Base cylinder along +Z
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    # Align cylinder direction to vector AB
    z_axis = np.array([0, 0, 1])
    if not np.allclose(vec / height, z_axis):
        R = rotation_between(np.array([0, 0, height]), vec)
    else:
        R = np.eye(4)

    # Translate to midpoint between A and B
    midpoint = (A + B) / 2
    T = translation_matrix(midpoint)

    # Apply transformation
    cyl.apply_transform(T @ R)
    return cyl



def rotation_between(vec1, vec2):
    """Compute a rotation matrix that rotates vec1 to vec2."""
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    axis = np.cross(vec1, vec2)
    if np.linalg.norm(axis) < 1e-8:
        # Already aligned or opposite
        if np.dot(vec1, vec2) > 0:
            return np.eye(4)
        else:
            # 180 degree rotation around any orthogonal axis
            axis = np.cross(vec1, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(vec1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
    else:
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        axis = axis / np.linalg.norm(axis)

    return trimesh.transformations.rotation_matrix(angle, axis)

