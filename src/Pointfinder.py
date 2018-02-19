import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
mtx = np.load('calibration/mtx.npy')
dist = np.load('calibration/dist.npy')
MIN_MATCH_COUNT = 40

cap = cv.VideoCapture(0)
surf = cv.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)
img2 = None
des2 = None
kp2 = None

while(1):
    _, frame = cap.read()
    h,w,_ = frame.shape
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
    
    k = cv.waitKey(5) & 0xFF
    
    if k == 32:
        img1 = frame
        kp1, des1 = surf.detectAndCompute(img1,None)
        if img2 is not None:
            matches = flann.knnMatch(des1,des2,k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            
            
            if len(matches)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                np.save('source_points',src_pts)
                np.save('destination_points',dst_pts)
                np.save('matrix',M)
                matchesMask = mask.ravel().tolist()
                h,w,_ = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts,M)
                img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = None, # draw only inliers
                           flags = 2)
                img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
                cv.imshow('frame3',img3)
                
        
        img2 = img1
        kp2 = kp1
        des2 = des1
        
    
    cv.imshow('frame',frame)
 

    if k == 27:
        break
cv.destroyAllWindows()