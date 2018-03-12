import numpy as np
import cv2
import time
import sys
import os
from sys import stdin


path = 'default'
camID = 0
if len(sys.argv) >= 3:
    path = sys.argv[2]
    camID = int(sys.argv[1])
else:
    print("Camera Name:")
    path = stdin.readline().rstrip('\n')
    print("Camera ID:")
    camID = int(stdin.readline())

if not os.path.exists(path):
    print("No calibration matrix found!")
    exit()
    
if os.path.exists(path + '/mtx.npy'):
    mtx = np.load(path + '/mtx.npy')
    dist = np.load(path + '/dist.npy')


md = False
def callback(event,x,y,flags,param):
    global md
    md = event == cv2.EVENT_LBUTTONDOWN

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',callback)
cap = cv2.VideoCapture(camID)


while(1):
    k = cv2.waitKey(5) & 0xFF
    
    _, frame = cap.read()
    w, h, _ = frame.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
     
   # x,y,w,h = roi
   # frame = frame[y:y+h, x:x+w]


    if k == 32 or md:
        cv2.imwrite("images/"+path+str(time.time())+".png" ,frame)
        md = False
        print('capture')
    
    cv2.imshow('frame', frame)
    if k == 27:
        break
    
cv2.destroyAllWindows()
exit()