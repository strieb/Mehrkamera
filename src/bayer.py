import numpy as np
import cv2
import homography as ho

cap1 = cv2.VideoCapture("videos/Trial04.61972048.avi")
cap2 = cv2.VideoCapture("videos/Trial04.64458576.avi")
path = "VICON"
mtx = np.load(path + '/mtx.npy')
dist = np.load(path + '/dist.npy')
_, frame1 = cap1.read()
_, frame2 = cap2.read()
w, h, _ = frame1.shape
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

box = np.asarray([[1, 1], [1, -1], [-1, -1], [-1, 1]])

frame1 = cv2.cvtColor(frame1[:, :, 0], cv2.COLOR_BayerGR2BGR)
frame1 = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
ref1 = frame1

frame2 = cv2.cvtColor(frame2[:, :, 0], cv2.COLOR_BayerGR2BGR)
frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
ref2 = frame2

#ref = cv2.imread("images/graffiti/ref.png")

#box = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])
#ref = cv2.resize(ref, (0, 0), fx=0.1, fy=0.1)
#box /= 10

#kp_r, des_r = ho.find_keypoints(ref)

hsv = np.zeros_like(frame1)
hsv[...,1] = 255

with ho.Graph() as graph:
    t = 0
    while(1):
        for i in range(0, 10):
            _, frame1 = cap1.read()
            _, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame1 = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
        frame2 = cv2.cvtColor(frame2[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
        t += 1
        if t %10 == 0:
            cv2.imwrite("images/stereo1-"+str(t)+".png",frame1)
            cv2.imwrite("images/stereo2-"+str(t)+".png",frame2)

        diff1 = cv2.absdiff(ref1,frame1)
        diff1 = cv2.cvtColor(diff1,cv2.COLOR_BGR2GRAY)
        diff1 = cv2.GaussianBlur(diff1,(9,9),0)
        _, diff1 = cv2.threshold(diff1, 16, 255, cv2.THRESH_BINARY)
        cv2.imshow('diff1', diff1)

        diff2 = cv2.absdiff(ref2,frame2)
        diff2 = cv2.cvtColor(diff2,cv2.COLOR_BGR2GRAY)
        diff2 = cv2.GaussianBlur(diff2,(9,9),0)
        _, diff2 = cv2.threshold(diff2, 16, 255, cv2.THRESH_BINARY)
        cv2.imshow('diff2', diff2)
        
        kp_1, des_1 = ho.find_keypoints(frame1, diff1)
        kp_2, des_2 = ho.find_keypoints(frame2, diff2)

        # pts_r, pts_1, good = ho.match_keypoints(kp_r, des_r, kp_1, des_1)
        # if pts_1.shape[0] > 20:
        #     H_r, mask_r = ho.findHomography(graph, pts_r, pts_1, 3)
        #     #box1 = ho.project(H_r, box)
        #     #frame1 = ho.box(box1, frame1)
        #     frame3 = ho.matches( kp_r, kp_1,good,  ref, frame1,None)
        #     cv2.imshow('frame3', frame3)
        if len(kp_1) > 20 and len(kp_2) > 20:
            pts_1, pts_2, good = ho.match_keypoints(kp_1, des_1, kp_2, des_2)
            if pts_1.shape[0] > 20:
                H, mask = ho.findHomography( pts_1, pts_2, 3, graph = graph)
                if mask.sum() > 20:
                    N = ho.findNormalizationMatrix(pts_1, mask)
                    N_inv = np.linalg.inv(N)

                    # box1 = pts1[mask][:30:10]
                    # box2 = ho.project(H,box1)

                    # frame1 = ho.box(box1,frame1)
                    # frame2 = ho.box(box2,frame2)

                    frame3 = cv2.warpPerspective(frame1, H, (frame1.shape[1], frame1.shape[0]))
                    frame4 = frame3/512 + frame2/512
                    cv2.imshow('frame4', frame4)

                    # frameA = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                    # frameB = cv2.cvtColor(frame3,cv2.COLOR_BGR2GRAY)
                    # flow = cv2.calcOpticalFlowFarneback(frameA,frameB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    # hsv[...,0] = ang*180/np.pi/2
                    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) * 100
                    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    # cv2.imshow('flow',bgr)


        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)

        # cv2.imshow('ref', ref)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
exit()
