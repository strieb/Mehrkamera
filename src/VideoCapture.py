import cv2
import numpy as np
import sys
import os

def saveVideo(file, camID, length, mtx, dist):
    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_FPS, 30)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps={:f} w={:f} h={:f}'.format(fps, w, h))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file+'.avi', fourcc, fps, (w, h))

    if mtx is not None:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    for i in range(1, length * 30):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            out.write(frame)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(5)
            if key == 27:  # exit on ESC
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

path = sys.argv[4]

if os.path.exists(path + '/mtx.npy'):
    mtx = np.load(path + '/mtx.npy')
    dist = np.load(path + '/dist.npy')


saveVideo(sys.argv[2], int(sys.argv[1]), int(sys.argv[3]), mtx, dist)
