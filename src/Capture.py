import numpy as np
import cv2
import time

md = False
def callback(event,x,y,flags,param):
    global md
    md = event == cv2.EVENT_LBUTTONDOWN

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',callback)
cap = cv2.VideoCapture(0)

mtx = np.matrix([[531.01819623, 0, 329.54092235], [0, 530.99003943, 265.93536334], [0, 0, 1]])
dist = np.matrix([[0.06322733, -0.21165693, 0.00498478, -0.00476949, 0.13033501]])


while(1):
    k = cv2.waitKey(5) & 0xFF
    
    _, frame = cap.read()
    frame = cv2.flip(frame, -1)
    w, h, _ = frame.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
     
    if k == 32 or md:
        cv2.imwrite(str(time.time())+".png" ,frame)
        md = False
        print('capture')
    
    cv2.imshow('frame', frame)
    if k == 27:
        break
    
cv2.destroyAllWindows()