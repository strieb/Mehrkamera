import numpy as np
import cv2 as cv
import time


MIN_MATCH_COUNT = 20
img1 = cv.imread('key1.png', 1)  # queryImage
cap = cv.VideoCapture(0)

# Initiate SIFT detector
fast = cv.FastFeatureDetector_create()


orb = cv.ORB_create()
# find the keypoints and descriptors with SIFT
kp1 = fast.detect(img1, None)
kp1, des1 = orb.compute(img1, kp1, None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)


bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

while(1):
    _, img2 = cap.read()

    k = cv.waitKey(5) & 0xFF
    if k == 32:
        height, width, d = img2.shape
        img1 = img2[int(height * 0.25):int(height * 0.75),
                    int(width * 0.25):int(width * 0.75)]
        kp1 = fast.detect(img1, None)
        kp1, des1 = orb.compute(img1, kp1, None)

    t = time.time()

    kp2 = fast.detect(img2, None)
    kp2, des2 = orb.compute(img2, kp2, None)

    print("orb " + str(t - time.time()))
    t = time.time()

    if des2 is not None:
        matches = bf.match(des1, des2)
        #matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
       # good = []
       # for m,n in matches:
        #    if m.distance < 0.7*n.distance:
        #        good.append(m)
       # matches = good

        print("flan " + str(t - time.time()))
        t = time.time()
        # store all the good matches as per Lowe's ratio test.

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            print(M)
            matchesMask = mask.ravel().tolist()
            h, w, _ = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
            print(
                "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(img1, kp1, img2, kp2,
                              matches, None, **draw_params)

        print("else " + str(t - time.time()))

        cv.imshow('frame', img3)

    if k == 27:
        break
cv.destroyAllWindows()
