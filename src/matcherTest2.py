import numpy as np
import cv2
import homography as ho

np.set_printoptions(suppress=True) 

if __name__ == "__main__":
    
    frame1 = cv2.imread("images/graffiti/2.png")
    frame2 = cv2.imread("images/graffiti/5.png")
    pts1, pts2, kp1, kp2, good = ho.match(frame1,frame2)

    H, mask = ho.ransac(pts1,pts2,5)
    print(np.sum(mask))
    src = pts1[mask]
    dst = pts2[mask]


    print(H)
    frame = frame1.copy()
    for x,y in src:
        cv2.circle(frame,(x,y),1,(255,0,0),1,cv2.LINE_AA)

    cv2.circle(frame,(mid[0],mid[1]),int(dist*1.5),(0,255,0),1,cv2.LINE_AA)

    while True:
        cv2.imshow("test", frame)   
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break