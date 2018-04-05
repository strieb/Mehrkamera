import numpy as np
from .. import homography as ho
from typing import Tuple


def ransac(src, dst, threshold) -> Tuple[np.ndarray, np.ndarray]:
    """Basic implementation of RANSAC.
    Returns:
        (H, mask) tuple of the homography and mask.
    """
    best = 0
    mask = None
    H = None
    n = src.shape[0]
    for i in range(0, 1000):
        choices = np.random.choice(n, 4)
        src_slice = np.take(src, choices, axis=0)
        dst_slice = np.take(dst, choices, axis=0)
        H_new = ho.findSVD(src_slice, dst_slice)
        error = ho.distanceError(H_new, src, dst)
        mask_new = error < threshold
        if mask_new.sum() > best:
            mask = error < threshold
            best = mask_new.sum()
            H = H_new
    return H, mask


def msac(src, dst, threshold) -> Tuple[np.ndarray, np.ndarray]:
    """Basic implementation of MSAC. https://de.wikipedia.org/wiki/RANSAC-Algorithmus#MSAC.
    Returns:
        (H, mask) tuple of the homography and mask.
    """
    best = 10000000
    mask = None
    H = None
    n = src.shape[0]
    for i in range(0, 1000):
        choices = np.random.choice(n, 4)
        src_slice = np.take(src, choices, axis=0)
        dst_slice = np.take(dst, choices, axis=0)
        H_new = ho.findSVD(src_slice, dst_slice)
        error = ho.distanceError(H_new, src, dst)
        mask_new = np.clip(error, 0, threshold)
        if mask_new.sum() < best:
            mask = error < threshold
            best = mask_new.sum()
            H = H_new
    return H, mask
