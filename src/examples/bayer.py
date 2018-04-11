'''Finds a homography between two synchronized videos that is not in the background. Keypoints in the background get excluded'''
import numpy as np
import cv2
import modules.homography as ho

cap1 = cv2.VideoCapture("res/videos/Trial04.61972048.avi")
cap2 = cv2.VideoCapture("res/videos/Trial04.64458576.avi")
path = "res/calibration/VICON"
mtx = np.load(path + '/mtx.npy')
dist = np.load(path + '/dist.npy')
_, frame1 = cap1.read()
_, frame2 = cap2.read()
w, h, _ = frame1.shape
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# The first frame is used as background
frame1 = cv2.cvtColor(frame1[:, :, 0], cv2.COLOR_BayerGR2BGR)
frame1 = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
ref1 = frame1

frame2 = cv2.cvtColor(frame2[:, :, 0], cv2.COLOR_BayerGR2BGR)
frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
ref2 = frame2

STEP_SIZE = 10

with ho.Graph() as graph:
    # t = 0
    while(1):
        for i in range(0, STEP_SIZE):
            _, frame1 = cap1.read()
            _, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame1 = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
        frame2 = cv2.cvtColor(frame2[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)
        # t += 1
        # if t % 10 == 0:
        #     cv2.imwrite("images/stereo1-"+str(t)+".png", frame1)
        #     cv2.imwrite("images/stereo2-"+str(t)+".png", frame2)


        # Calculate difference between frames and background and use it as a mask for keypoint detection
        diff1 = cv2.absdiff(ref1, frame1)
        diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
        diff1 = cv2.GaussianBlur(diff1, (9, 9), 0)
        _, diff1 = cv2.threshold(diff1, 16, 255, cv2.THRESH_BINARY)
        cv2.imshow('diff1', diff1)

        diff2 = cv2.absdiff(ref2, frame2)
        diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
        diff2 = cv2.GaussianBlur(diff2, (9, 9), 0)
        _, diff2 = cv2.threshold(diff2, 16, 255, cv2.THRESH_BINARY)
        cv2.imshow('diff2', diff2)

        kp_1, des_1 = ho.findKeypoints(frame1, diff1)
        kp_2, des_2 = ho.findKeypoints(frame2, diff2)

        if len(kp_1) > 20 and len(kp_2) > 20:
            pts_1, pts_2, good = ho.matchKeypoints(kp_1, des_1, kp_2, des_2)
            if pts_1.shape[0] >= 10:
                H, mask = ho.findHomography(pts_1, pts_2, 3, graph=graph)
                if mask.sum() >= 10:
                    frame1Warp = cv2.warpPerspective(frame1, H, (frame1.shape[1], frame1.shape[0]))
                    frameBlend = frame1Warp/256 * 0.5 + frame2/256 * 0.5
                    cv2.imshow('frame4', frameBlend)

        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
exit()
