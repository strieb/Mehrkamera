import numpy as np
import homography as ho

def ransac(src,dst, threshold):
    # threshold = threshold * threshold * 2
    best = 0
    mask = None
    H = None
    n = src.shape[0]
    for i in range(0,1000):
        choices = np.random.choice(n,4)
        src_slice = np.take(src,choices,axis = 0)
        dst_slice = np.take(dst,choices,axis = 0)
        H_new = ho.findSVD(src_slice,dst_slice)
        error = ho.distance_error(H_new,src,dst)
        mask_new = error < threshold
        if mask_new.sum() > best:
            mask = error < threshold
            best = mask_new.sum()
            H = H_new
    return H, mask


def ransac2(src,dst, threshold):
    best = 10000000
    mask = None
    H = None
    n = src.shape[0]
    for i in range(0,1000):
        choices = np.random.choice(n,4)
        src_slice = np.take(src,choices,axis = 0)
        dst_slice = np.take(dst,choices,axis = 0)
        H_new = ho.findSVD(src_slice,dst_slice)
        error = ho.distance_error(H_new,src,dst)
        mask_new = np.clip(error,0,threshold*2)
        if mask_new.sum() < best:
            mask = error < threshold
            best = mask_new.sum()
            H = H_new
    return H, mask
