import cv2
import numpy as np
import sys

def saveVideo(file, camID):
    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_FPS, 30)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps={:f} w={:f} h={:f}'.format(fps,w,h))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file+'.avi',fourcc, fps, (int(w),int(h)))
    
    while True:
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            cv2.imshow("frame",frame)
            key = cv2.waitKey(5)
            if key == 27:  # exit on ESC
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


saveVideo(sys.argv[2], int(sys.argv[1]))