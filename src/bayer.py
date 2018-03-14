import numpy as np
import cv2
import homography as ho

cap1 = cv2.VideoCapture("videos/Trial03.61972048.avi")
cap2 = cv2.VideoCapture("videos/Trial03.64458576.avi")
path = "VICON"
mtx = np.load(path + '/mtx.npy')
dist = np.load(path + '/dist.npy')
_, frame1 = cap1.read()
_, frame2 = cap2.read()
w, h, _ = frame1.shape
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

box = np.asarray([[1, 1], [1, -1], [-1, -1], [-1, 1]])

# ref = cv2.imread("images/graffiti/ref.png")
# box = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]]).reshape(-1, 1, 2)
# box1 = box
# ref = cv2.resize(ref, (0, 0), fx=0.1, fy=0.1)
# box /= 10

with ho.Graph() as graph:
    while(1):
        for i in range(0, 20):
            _, frame1 = cap1.read()
            _, frame2 = cap2.read()
        frame1 = cv2.cvtColor(frame1[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame1 = cv2.undistort(frame1, mtx, dist, None, newcameramtx)
        frame2 = cv2.cvtColor(frame2[:, :, 0], cv2.COLOR_BayerGR2BGR)
        frame2 = cv2.undistort(frame2, mtx, dist, None, newcameramtx)

        # pts_r, pts1, kp_r, kp1, good = ho.match(ref, frame1)
        # if pts1.shape[0] > 20:
        #     H_r, mask_r = ho.findHomography(graph, pts_r, pts1, 3)
        #     box1 = cv2.perspectiveTransform(box,H_r)
        #     frame1_x = ho.box(H_r,box,frame1)
        #     cv2.imshow('frame1', frame1_x)

        pts1, pts2, kp1, kp2, good = ho.match(frame1, frame2)
        if pts1.shape[0] > 20:
            H, mask = ho.findHomography(graph, pts1, pts2, 3)
            if mask.sum() > 20:
                N = ho.findNormalizationMatrix(pts1, mask)
                N_inv = np.linalg.inv(N)

                # box1 = pts1[mask][:30:10]
                # box2 = ho.project(H,box1)

                # frame1 = ho.box(box1,frame1)
                # frame2 = ho.box(box2,frame2)

                frame3 = cv2.warpPerspective(frame1, H, (frame1.shape[1],frame1.shape[0]))
                frame4 = frame3/512 + frame2/512
                cv2.imshow('frame4', frame4)

        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)

        # cv2.imshow('ref', ref)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
exit()
