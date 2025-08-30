import numpy as np 
import cv2


def find_bbox(keypoints): 
    bbox = np.zeros((4,), dtype=np.int32)
    x_min = np.min(keypoints[:, 0])
    x_max = np.max(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    y_max = np.max(keypoints[:, 1])
    bbox[0] = int(x_min)
    bbox[1] = int(y_min)
    bbox[2] = int(x_max - x_min)
    bbox[3] = int(y_max - y_min)
    return bbox

def drawBBox(img, keypoints, color):
    bbox = find_bbox(keypoints)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
    return img

def drawKeypoints(img, keypoints, color):
    for keypoint in keypoints:
        cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), 2, color, -1)
    return img

# From SMPL FAQ
KINEMATIC_TREE = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]

def drawSkeleton(img, joints, keypointColor, lineColor):
    for i, j in KINEMATIC_TREE:
        cv2.line(img, (int(joints[i][0]), int(joints[i][1])), (int(joints[j][0]), int(joints[j][1])), lineColor, 2)
    for joint in joints:
        cv2.circle(img, (int(joint[0]), int(joint[1])), 2, keypointColor, -1)
    return img

def exportMesh(savepath, vertices, faces):
    with open(savepath, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for face in faces:
            f.write(f'f {" ".join(str(v + 1) for v in face)}\n')

def drawMesh(img, vertices, faces, lineColor):
    img_height, img_width = img.shape[:2]
    verts = vertices.copy()
    # normalize vertices
    verts[:, 0] = (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min())
    verts[:, 1] = (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].max() - verts[:, 1].min())
    # Project vertices to image coordinates
    verts2d = np.zeros((verts.shape[0], 2), dtype=np.int32)
    verts2d[:, 0] = (verts[:, 0] * img_width).astype(np.int32)
    verts2d[:, 1] = (verts[:, 1] * img_height).astype(np.int32)

    # Draw faces
    for face in faces:
        pts = verts2d[face]
        cv2.fillPoly(img, [pts], color=lineColor)

    return img