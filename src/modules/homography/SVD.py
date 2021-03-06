import numpy as np
from typing import Tuple


def findNormalizationMatrixWithSize(size: Tuple[int, int]) -> np.ndarray:
    dim = max(size[0], size[1])
    return np.asarray([[2/dim, 0., -size[0]/dim], [0., 2/dim, -size[1]/dim], [0., 0., 1.]])


def findNormalizationMatrix(pts, mask=None, factor=2) -> np.ndarray:
    if mask is not None:
        if mask.dtype != np.bool:
            raise Exception("Mask is not a boolean.")
        pts = pts[mask]
    mid = pts.mean(axis=0)
    dist = np.linalg.norm(pts-mid, axis=1).mean()
    scale = 1/(factor*dist)
    return np.asarray([[1*scale, 0., -mid[0]*scale],
                       [0., 1*scale, -mid[1]*scale],
                       [0., 0., 1.]])


def findSVD(src, dst, ids=[0, 1, 2, 3]) -> np.ndarray:
    A = np.zeros((0, 9))
    for i in ids:
        A = np.concatenate((A, _createCorrespondanceMatrix(src[i], dst[i])), axis=0)
    u, s, vh = np.linalg.svd(A)
    M2 = np.reshape(vh[-1] / vh[-1, -1], (3, 3))
    return M2


def distanceError(H, src, dst, mask=None) -> np.ndarray:
    if mask is not None:
        if mask.dtype != np.bool:
            raise Exception("Mask is not a boolean.")
        src = src[mask]
        dst = dst[mask]

    scr_proj = project(H, src)
    err = np.linalg.norm(dst-scr_proj, axis=1)
    return err


def goldStandardError(H, src, dst, mask=None) -> np.ndarray:

    if mask is not None:
        if mask.dtype != np.bool:
            raise Exception("Mask is not a boolean.")
        src = src[mask]
        dst = dst[mask]

    H_inv = np.linalg.inv(H)
    scr_proj = project(H, src)
    dst_proj = project(H_inv, dst)
    err = np.linalg.norm(dst-scr_proj, axis=1)**2 + np.linalg.norm(src-dst_proj, axis=1)**2
    return err


def _createCorrespondanceMatrix(src, dst):
    A = np.zeros((2, 9))
    A[0] = [-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[0], src[1] * dst[0], dst[0]]
    A[1] = [0, 0, 0, -src[0], -src[1], -1, src[0] * dst[1], src[1] * dst[1], dst[1]]
    return A


def project(H, b) -> np.ndarray:
    if b.ndim == 2:
        if b.shape[1] == 2:
            bnew = np.ones((b.shape[0], b.shape[1]+1))
            bnew[:, :2] = b
            b = bnew
        a = np.matmul(b, H.transpose())
        a[:, 0] /= a[:, 2]
        a[:, 1] /= a[:, 2]
        return a[:, :2]
    else:
        if b.shape[0] == 2:
            b = np.append(b, 1)
        a = np.matmul(H, b)
        a /= a[2]
        return a[:2]
