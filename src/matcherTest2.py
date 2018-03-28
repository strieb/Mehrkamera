import numpy as np
import cv2
import homography as ho

np.set_printoptions(suppress=True)

if __name__ == "__main__":

    frame1 = cv2.imread("images/graffiti/ref.png")
    pts = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])
    frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)
    pts /= 4

    frame2 = cv2.imread("images/graffiti/13.jpg")
    frame2 = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
    pts1, pts2, kp1, kp2, good = ho.match(frame1, frame2)

    src = pts1
    dst = pts2

    with ho.Graph() as graph:
        H, mask = ho.findHomography(graph, src, dst, 4, epochs=0, learning_rate=0.2)

    frame3 = cv2.warpPerspective(frame2, np.linalg.inv(H), (frame1.shape[1], frame1.shape[0]))
    frame4 = frame3/512 + frame1/512
    diff = cv2.absdiff(frame3, frame1)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame3', frame3)
    cv2.imshow('frame4', frame4)
    cv2.imshow('diff', diff)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frameA = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    frameB = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frameA, frameB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', bgr)

    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
