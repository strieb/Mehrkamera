import numpy as np
import cv2


def findHomography(src, dst):
    A = np.zeros((0, 9))
    for i in range(0, 40,10):
        A = np.concatenate((A, createCorrespondanceMatrix(src[i],dst[i])), axis=0)
    u, s, vh = np.linalg.svd(A)
    M2 = np.reshape(vh[-1] / vh[-1, -1], (3, 3))
    return M2


def createCorrespondanceMatrix(src,dst):
    A = np.zeros((2,9))
    A[0] = [-src[0], -src[1], -1, 0, 0, 0, src[0] * dst[0], src[1] * dst[0], dst[0]]
    A[1] = [0, 0, 0, -src[0], -src[1], -1, src[0] * dst[1], src[1] * dst[1], dst[1]]
    return A

def insertPoint(A, src, dst, i):
    A[i*2, 0] = -src[0]
    A[i*2, 1] = -src[1]
    A[i*2, 2] = -1
    A[i*2, 6] = src[0] * dst[0]
    A[i*2, 7] = src[1] * dst[0]
    A[i*2, 8] = dst[0]
    A[i*2 + 1, 3] = -src[0]
    A[i*2 + 1, 4] = -src[1]
    A[i*2 + 1, 5] = -1
    A[i*2 + 1, 6] = src[0] * dst[1]
    A[i*2 + 1, 7] = src[1] * dst[1]
    A[i*2 + 1, 8] = dst[1]


def project(M, b):
    if b.shape == (2,):
        b = np.append(b, 1)
    a = np.matmul(M, b)
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

    M = findHomography(src, dst)

    for i in range(10,20):
        b = src[i]
        a1 = dst[i]
        a2 = project(M,b)
        print(np.linalg.norm(a1-a2))
