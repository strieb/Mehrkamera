import numpy as np
import cv2
import homography as ho

np.set_printoptions(suppress=True)


class ImageWithPoints:
    def __init__(self, image, points):
        self.image = image
        self.points = points
        kp, des = ho.find_keypoints(image)
        self.kp = kp
        self.des = des


def compare(img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.match_keypoints(img1.kp, img1.des, img2.kp, img2.des)
    H_tf_, mask_tf = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=20,learning_rate=0.3)
    H_tf, mask_tf = ho.findHomography(pts_match_1, pts_match_2, 4)
    H_cv, mask_cv = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

    results = np.zeros((2, 4))

    results[0][0] = ho.distance_error(H_tf, pts_match_1, pts_match_2, mask_tf).mean()
    results[0][1] = ho.distance_error(H_tf, pts_match_1, pts_match_2, mask_cv).mean()
    results[0][2] = ho.distance_error(H_tf, img1.points, img2.points, None).mean()
    results[0][3] = np.sum(mask_tf)

    results[1][0] = ho.distance_error(H_cv, pts_match_1, pts_match_2, mask_tf).mean()
    results[1][1] = ho.distance_error(H_cv, pts_match_1, pts_match_2, mask_cv).mean()
    results[1][2] = ho.distance_error(H_cv, img1.points, img2.points, None).mean()
    results[1][3] = np.sum(mask_cv)

    return results


def compareAndShow(img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.match_keypoints(img1.kp, img1.des, img2.kp, img2.des)
    H_tf, mask_tf = ho.findHomography(pts_match_1, pts_match_2, 4)
    H_cv, mask_cv = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

    image = ho.box(img2.points, img2.image, color=(0, 0, 255))

    image_tf = ho.box(ho.project(H_tf, img1.points), image, color=(255, 0, 0))
    image_cv = ho.box(ho.project(H_cv, img1.points), image, color=(0, 255, 0))

    while True:
        cv2.imshow('tf', image_tf)
        cv2.imshow('cv', image_cv)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break


if __name__ == "__main__":

    frame = cv2.imread("images/graffiti/ref.png")
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    pts = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])
    pts /= 4

    image1 = ImageWithPoints(frame, pts)

    frame = cv2.imread("images/graffiti/14.jpg")
    pts = np.float32([[1335, 452],  [2819, 1061], [2391, 1887], [1137, 1336]])
    pts /= 4
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    image2 = ImageWithPoints(frame, pts)

    frame = cv2.imread("images/graffiti/10.png")
    pts = np.float32([[336, 330],  [443, 182], [616, 317], [452, 439]])
    image3 = ImageWithPoints(frame, pts)

    frame = cv2.imread("images/graffiti/stereo1-100.png")
    pts = np.float32([[296,117],[463,84],[479,205],[323,250]])
    stereo1 = ImageWithPoints(frame, pts)
    frame = cv2.imread("images/graffiti/stereo2-100.png")
    pts = np.float32([[99,94],[289,66],[311,196],[129,225]])
    stereo2 = ImageWithPoints(frame, pts)

    frame = cv2.imread("images/graffiti/stereo1-60.png")
    pts = np.float32([[317,151],[481,150],[478,269],[318,262]])
    stereo3 = ImageWithPoints(frame, pts)
    frame = cv2.imread("images/graffiti/stereo2-60.png")
    pts = np.float32([[179,131],[312,137],[306,262],[177,240]])
    stereo4 = ImageWithPoints(frame, pts)

    #pts = np.float32([[,],[,],[,],[,]])

    print(compare(image1, image2))
    print(compare(image1, image3))
    print(compare(image2, image3))
    print(compare(stereo1, stereo2))
    print(compare(stereo3, stereo4))

    compareAndShow(image1,image3)
