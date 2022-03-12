import cv2
import numpy as np


def calculate_scale(bbox, scale_Factor, size):
    width = bbox[2]
    height = bbox[3]

    if width > height:
        height = width
    if height > width:
        width = height

    scale = np.array([width / float(size), height / float(size)], dtype=np.float)
    scale = scale_Factor * scale

    return scale


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_transforms(bbox, scale_factor, rotation_factor, output_size, shift_factor=np.array([0, 0], dtype=np.float32),
                   inv=False):

    scale_size = calculate_scale(bbox, scale_factor, output_size)

    scale_tmp = scale_size * output_size
    src_w = scale_tmp[0]
    dst_w = output_size
    dst_h = output_size

    src_dir = get_dir([0, src_w * -0.5], rotation_factor)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    center = np.array([bbox[0] + (bbox[2] / 2.0), bbox[1] + (bbox[3] / 2.0)], dtype=np.float)

    src[0, :] = center + scale_tmp * shift_factor
    src[1, :] = center + src_dir + scale_tmp * shift_factor
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, trans):

    coords = coords - trans[:, 2]
    coords = coords.T
    coords = np.dot(np.linalg.inv(trans[:, 0:2]), coords)
    return coords.T
