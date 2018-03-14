import numpy as np
import cv2

if __name__ == "__main__":
    
    frame = cv2.imread("images/keyboard/key3.png")
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
    frame = cv2.blur(frame,(3,3))
    frame = cv2.blur(frame,(3,3))

    while True:
        cv2.imshow("test", frame)   

        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
