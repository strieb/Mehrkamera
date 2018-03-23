import numpy as np
import cv2 as cv

MIN_MATCH_COUNT = 4

surf = cv.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)


def find_keypoints(frame, mask = None):
    kp, des = surf.detectAndCompute(frame, mask)
    return kp, des

def match_keypoints(kp1,des1,kp2,des2):
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(matches) > MIN_MATCH_COUNT:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return pts1, pts2, good

def match(frame1, frame2):
    kp1, des1 = surf.detectAndCompute(frame1, None)
    kp2, des2 = surf.detectAndCompute(frame2, None)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(matches) > MIN_MATCH_COUNT:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return pts1, pts2, kp1, kp2, good



def box(pts,img, color = (255,0,0)):
    dst = pts.reshape(-1, 1, 2)
    imgb = img.copy()
    cv.polylines(imgb, [np.int32(dst)], True, color, 1, cv.LINE_AA)
    return imgb

def matches(kp1, kp2, good, img1,img2, mask):
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=mask,  # draw only inliers
                        flags=2)
    img = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img