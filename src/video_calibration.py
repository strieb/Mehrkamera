import numpy as np
import cv2
import sys
from sys import stdin
import os

path = 'default'
camID = 0
if len(sys.argv) >= 3:
    path = sys.argv[2]
    camID = str(sys.argv[1])
else:
    print("Camera Name:")
    path = stdin.readline().rstrip('\n')
    print("Video:")
    camID = str(stdin.readline()).rstrip()

if not os.path.exists(path):
    os.makedirs(path)

mtx = np.matrix([[531.01819623, 0, 329.54092235], [0, 530.99003943, 265.93536334], [0, 0, 1]])
dist = np.matrix([[0.06322733, -0.21165693, 0.00498478, -0.00476949, 0.13033501]])

if os.path.exists(path + '/mtx.npy'):
    mtx = np.load(path + '/mtx.npy')
    dist = np.load(path + '/dist.npy')

cap = cv2.VideoCapture(camID)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
k = -1

while(1):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_BayerGR2BGR)
    
    if k == 32:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            frame = cv2.drawChessboardCorners(frame, (9, 6), corners2, ret)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print("added")
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    w, h, _ = frame.shape;
    if k == 107:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('calibrated')
        np.save(path+'/mtx',mtx)
        np.save(path+'/dist',dist)
        print(mtx)
        print(dist)
        
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    cv2.imshow('undist', dst)
    
cv2.destroyAllWindows()
