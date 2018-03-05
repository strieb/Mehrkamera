import numpy as np
import cv2

def findNormalizationMatrixWithSize(size):
    dim = max(size[0],size[1])
    return np.asarray([[2/dim,0.,-size[0]/dim],[0.,2/dim,-size[1]/dim],[0.,0.,1.]])


def findSVD(src, dst, ids = [0,1,2,3]):
    A = np.zeros((0, 9))
    for i in ids:
        A = np.concatenate((A, _createCorrespondanceMatrix(src[i],dst[i])), axis=0)
    u, s, vh = np.linalg.svd(A)
    M2 = np.reshape(vh[-1] / vh[-1, -1], (3, 3))
    return M2

def goldStandardError(H, src, dst, mask = None):
    if mask is not None:
        src = src[mask]
        dst = dst[mask]

    H_inv = np.linalg.inv(H)
    scr_proj = project(H,src)
    dst_proj = project(H_inv,dst)
    err = np.linalg.norm(dst-scr_proj, axis=1)**2 + np.linalg.norm(src-dst_proj, axis=1)**2
    return err
    
def _createCorrespondanceMatrix(src,dst):
    A = np.zeros((2,9))
    A[0] = [-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[0], src[1] * dst[0], dst[0]]
    A[1] = [0, 0, 0, -src[0], -src[1], -1, src[0] * dst[1], src[1] * dst[1], dst[1]]
    return A

def project(H, b):
    if b.ndim == 2:
        if b.shape[1] == 2:
            bnew = np.ones((b.shape[0],b.shape[1]+1))
            bnew[:,:2] = b
            b = bnew
        a = np.matmul(b,H.transpose())
        a[:,0] /= a[:,2]
        a[:,1] /= a[:,2]
        return a[:,:2]
    else:
        if b.shape[0] == 2:
            b = np.append(b, 1)
        a = np.matmul(H, b)
        a /= a[2]
        return a[:2]


if __name__ == "__main__":
    src = np.load('source_points.npy')
    dst = np.load('destination_points.npy')
    #src = np.asarray([[0,0],[0,10],[10,10],[10,0],[0,5],[5,5]])
    #dst = np.asarray([[1,1],[1,21],[11,21],[11,1],[1,11],[6,11]])

    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask = np.where(mask.transpose()[0] == 0)[0]
    src = np.delete(src,mask,0)
    dst = np.delete(dst,mask,0)

    M = findSVD(src, dst)

    for i in range(10,20):
        b = src[i]
        a1 = dst[i]
        a2 = project(M,b)
        print(np.linalg.norm(a1-a2))
