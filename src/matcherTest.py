import numpy as np
import cv2
import homography as ho

np.set_printoptions(suppress=True)


def draw(img, pts, H):
    h, w, _ = img.shape
    dst = cv2.perspectiveTransform(pts, H)
    imgb = img.copy()
    cv2.polylines(imgb, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    return imgb


if __name__ == "__main__":

    frame1 = cv2.imread("images/graffiti/ref.png")
    pts = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])
    frame1 = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)
    pts /= 4

    frame2 = cv2.imread("images/graffiti/14.jpg")
    frame2 = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
    pts1, pts2, kp1, kp2, good = ho.match(frame1, frame2)

    src = pts1
    dst = pts2

    # pts = np.float32([[81, 69], [418, 63], [420, 277], [113, 315]]).reshape(-1, 1, 2)

    with ho.Graph() as graph:
        H, mask = ho.findHomography(graph, src, dst, 3)

    H_cv2, mask_cv2 = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    mask_cv2 = mask_cv2.ravel()
    mask_cv2 = mask_cv2 > 0.5

    # H = np.matmul(np.linalg.inv(N),np.matmul(H,N))
    # H /= H[2,2]

    # H_cv2 = np.matmul(np.linalg.inv(N),np.matmul(H_cv2,N))
    # H_cv2 /= H_cv2[2,2]

    print(H)
    print(H_cv2)

    print("mask")
    print(np.sum(mask))
    print("mask_CV2")
    print(np.sum(mask_cv2))
    print("TF")
    print(ho.goldStandardError(H, pts1, pts2, mask).mean())
    print(ho.goldStandardError(H, pts1, pts2, mask_cv2).mean())
    print(ho.distanceError(H, pts1, pts2, mask).mean())
    print(ho.distanceError(H, pts1, pts2, mask_cv2).mean())
    print("CV2")
    print(ho.goldStandardError(H_cv2, pts1, pts2, mask).mean())
    print(ho.goldStandardError(H_cv2, pts1, pts2, mask_cv2).mean())
    print(ho.distanceError(H_cv2, pts1, pts2, mask).mean())
    print(ho.distanceError(H_cv2, pts1, pts2, mask_cv2).mean())

    mask_int = (mask * 1).tolist()
    while True:
        boxed = ho.polyline(ho.project(H, pts), frame2)
        boxed = ho.polyline(ho.project(H_cv2, pts), boxed)
        frame3 = ho.drawMatches(kp1, kp2, good, frame1, boxed, mask_int)
        cv2.imshow("test", frame3)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break
