import numpy as np
import homography as ho
import cv2

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    frame1 = cv2.imread("images/graffiti/2.png")
    frame2 = cv2.imread("images/graffiti/6.png")
    src, dst, kp1, kp2, good = ho.match(frame1,frame2)

    H, mask = ho.ransac(src, dst, 5)
    error = ho.goldStandardError(H,src,dst)
    mask = mask * 1


    H2, mask2 = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    mask2 = mask.ravel().tolist()

    print(H)
    print(H2)
    print(mask-mask2)
