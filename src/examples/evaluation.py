import numpy as np
import cv2
import modules.homography as ho

np.set_printoptions(suppress=True)
class ImageWithPoints:
    def __init__(self, image, points):
        self.image = image
        self.points = points
        kp, des = ho.findKeypoints(image)
        self.kp = kp
        self.des = des

def printResults(name, H, pts_src, pts_dst, mask):
    e1 = ho.distanceError(H, pts_src, pts_dst, mask).mean()
    e2 = ho.goldStandardError(H, pts_src, pts_dst, mask).mean()
    if mask is None:
        m = pts_src.shape[0]
    else:
        m = np.sum(mask)
    print("{:30s}|{:10.4f}|{:10.4f}|{:5.0f}".format(name, e1, e2, m))


def printHeader():
    print("{:30s}|{:10s}|{:10s}|{:5s}".format("image|method|mask", "distance error", "gold standard error", "# points"))

def compareLearningRate(name, img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)
    H, mask = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=0)
  
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=50, method=0, H=H, mask=mask, learning_rate=0.5)
    printResults(name+"|lr=0.3|200", H, pts_match_1, pts_match_2, mask)
    
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=300, method=0, H=H, mask=mask, learning_rate=0.5)
    printResults(name+"|lr=0.3|500", H, pts_match_1, pts_match_2, mask)
    
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=50, method=0, H=H, mask=mask, learning_rate=0.8)
    printResults(name+"|lr=0.5|200", H, pts_match_1, pts_match_2, mask)

    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=300, method=0, H=H, mask=mask, learning_rate=0.8)
    printResults(name+"|lr=0.5|500", H, pts_match_1, pts_match_2, mask)
    
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=50, method=0, H=H, mask=mask, learning_rate=1)
    printResults(name+"|lr=0.7|200", H, pts_match_1, pts_match_2, mask)
    
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=300, method=0, H=H, mask=mask, learning_rate=1)
    printResults(name+"|lr=0.7|500", H, pts_match_1, pts_match_2, mask)


def compareLearningRate2(name, img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)
    _, mask = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

  
    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=200, method=0, H=H, mask=mask, learning_rate=0.3)
    printResults(name+"|lr=0.3|200", H, pts_match_1, pts_match_2, mask)
    
    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=500, method=0, H=H, mask=mask, learning_rate=0.3)
    printResults(name+"|lr=0.3|500", H, pts_match_1, pts_match_2, mask)
    
    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=200, method=0, H=H, mask=mask, learning_rate=0.5)
    printResults(name+"|lr=0.5|200", H, pts_match_1, pts_match_2, mask)

    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=500, method=0, H=H, mask=mask, learning_rate=0.5)
    printResults(name+"|lr=0.5|500", H, pts_match_1, pts_match_2, mask)
    

    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=200, method=0, H=H, mask=mask, learning_rate=0.7)
    printResults(name+"|lr=0.7|200", H, pts_match_1, pts_match_2, mask)
    
    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    H, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=500, method=0, H=H, mask=mask, learning_rate=0.7)
    printResults(name+"|lr=0.7|500", H, pts_match_1, pts_match_2, mask)

def CompareRANSACWithout(name, img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)

    H_0, mask_0 = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=0,learning_rate=0.3)
    printResults(name+"|with", H_0, pts_match_1, pts_match_2, mask_0)
    error_ransac = ho.distanceError(H_0, pts_match_1, pts_match_2, mask_0).mean()

    H_0 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    z = 0
    with ho.Graph() as graph:
        while True:
            z += 10
            H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=10, learning_rate=0.3, method=0, H=H_0, mask=mask_0, graph = graph)
            error = ho.distanceError(H_0, pts_match_1, pts_match_2, mask_0).mean()
            if error < error_ransac:
                break
    printResults(name+"|without", H_0, pts_match_1, pts_match_2, mask_0)
    print("Epochs: "+str(z))


def compareWithout(name, img1: ImageWithPoints, img2: ImageWithPoints):
    '''Comparation with OpenCV. Both methods use the same keypoints. Our system does not use a prior SVD to find an estimation.'''
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)

    H_0 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=0,learning_rate=0.3,ransac=2)
    H_cv, mask_0 = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

    printResults(name+"|cv", H_cv, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|cv", H_cv, img1.points, img2.points, None)

    H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=100, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    printResults(name+"|100 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|100 epochs", H_0, img1.points, img2.points, None)
    # H_0 = H_0 / H_0[2, 2]
    # print(np.linalg.norm(H_0- H_cv))

    H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=100, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    # printResults(name+"|200 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|200 epochs", H_0, img1.points, img2.points, None)
    # H_0 = H_0 / H_0[2, 2]
    # print(np.linalg.norm(H_0- H_cv))

    H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=100, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    printResults(name+"|300 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|300 epochs", H_0, img1.points, img2.points, None)
    # H_0 = H_0 / H_0[2, 2]
    # print(np.linalg.norm(H_0- H_cv))

    H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=100, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    # printResults(name+"|400 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|400 epochs", H_0, img1.points, img2.points, None)
    # H_0 = H_0 / H_0[2, 2]
    # print(np.linalg.norm(H_0- H_cv))

    H_0, _ = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=100, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    printResults(name+"|500 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    # printResults(name+"|500 epochs", H_0, img1.points, img2.points, None)
    # H_0 = H_0 / H_0[2, 2]
    # print(np.linalg.norm(H_0- H_cv))


def compareEpochs(name, img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)

    H_0, mask_0 = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=0, learning_rate=0.3, method=2)
    H_1, mask_1 = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=2, learning_rate=0.3, method=0, H=H_0, mask=mask_0)
    H_2, mask_2 = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=8, learning_rate=0.3, method=0, H=H_1, mask=mask_0)
    H_3, mask_3 = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=10, learning_rate=0.3, method=0, H=H_2, mask=mask_0)
    H_4, mask_4 = ho.findHomography(pts_match_1, pts_match_2, 4, epochs=20, learning_rate=0.3, method=0, H=H_3, mask=mask_0)

    H_cv, mask_cv = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

    printResults(name+"|0 epochs", H_0, pts_match_1, pts_match_2, mask_0)
    printResults(name+"|2 epochs", H_1, pts_match_1, pts_match_2, mask_0)
    printResults(name+"|10 epochs", H_2, pts_match_1, pts_match_2, mask_0)
    printResults(name+"|20 epochs", H_3, pts_match_1, pts_match_2, mask_0)
    printResults(name+"|40 epochs", H_4, pts_match_1, pts_match_2, mask_0)
    printResults(name+"|0 epochs", H_0, img1.points, img2.points, None)
    printResults(name+"|2 epochs", H_1, img1.points, img2.points, None)
    printResults(name+"|10 epochs", H_2, img1.points, img2.points, None)
    printResults(name+"|20 epochs", H_3, img1.points, img2.points, None)
    printResults(name+"|40 epochs", H_4, img1.points, img2.points, None)
    printResults(name+"|cv", H_cv, img1.points, img2.points, None)


def compareRANSACMethod(name, img1: ImageWithPoints, img2: ImageWithPoints):
    '''Compares different methods for outlier detection.'''
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)
    H, mask = cv2.findHomography(pts_match_1, pts_match_2, cv2.RANSAC, 4)
    mask = mask.ravel() > 0.5
    # printResults(name+"|CV2-RANSAC", H, pts_match_1, pts_match_2, mask)
    printResults(name+"|CV2-RANSAC", H, img1.points, img2.points, None)

    H, mask = cv2.findHomography(pts_match_1, pts_match_2, cv2.LMEDS, 4)
    mask = mask.ravel() > 0.5
    # printResults(name+"|CV2-LMEDS", H, pts_match_1, pts_match_2, mask)
    printResults(name+"|CV2-LMEDS", H, img1.points, img2.points, None)

    H, mask = cv2.findHomography(pts_match_1, pts_match_2, cv2.RHO, 4)
    mask = mask.ravel() > 0.5
    # printResults(name+"|CV2-RHO", H, pts_match_1, pts_match_2, mask)
    printResults(name+"|CV2-RHO", H, img1.points, img2.points, None)

    H, mask = ho.ransac(pts_match_1, pts_match_2, 4)
    H, mask = ho.findHomographyCV(pts_match_1, pts_match_2, 0, mask=mask)
    # printResults(name+"|RANSAC", H, pts_match_1, pts_match_2, mask)
    printResults(name+"|RANSAC", H, img1.points, img2.points, None)

    H, mask = ho.msac(pts_match_1, pts_match_2, 4)
    H, mask = ho.findHomographyCV(pts_match_1, pts_match_2, 0, mask=mask)
    # printResults(name+"|MSAC", H, pts_match_1, pts_match_2, mask)
    printResults(name+"|MSAC", H, img1.points, img2.points, None)



def compareAndShow(img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)
    H_tf, mask_tf = ho.findHomography(pts_match_1, pts_match_2, 4)
    H_cv, mask_cv = ho.findHomographyCV(pts_match_1, pts_match_2, 4)

    image = ho.polyline(img2.points, img2.image, color=(0, 0, 255))

    image_tf = ho.polyline(ho.project(H_tf, img1.points), image, color=(255, 0, 0))
    image_cv = ho.polyline(ho.project(H_cv, img1.points), image, color=(0, 255, 0))

    while True:
        cv2.imshow('image', image)
        cv2.imshow('tf', image_tf)
        cv2.imshow('cv', image_cv)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break


def normalizationBox(img1, img2):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)

    pts = np.asarray([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    H, mask = ho.ransac(pts_match_1, pts_match_2, 4)
    N_1 = ho.findNormalizationMatrix(pts_match_1, mask)
    N_1_inv = np.linalg.inv(N_1)

    N_2 = ho.findNormalizationMatrix(pts_match_2, mask)
    N_2 = np.matmul(N_1, np.linalg.inv(H))
    N_2_inv = np.linalg.inv(N_2)

    image1 = ho.polyline(ho.project(N_1_inv, pts), img1.image)
    image2 = ho.polyline(ho.project(N_2_inv, pts), img2.image)

    mask = (mask * 1).tolist()
    both = ho.drawMatches(img1.kp, img2.kp, match_good, image1, image2, mask)

    cv2.imshow("both", both)
    cv2.waitKey(0)


def video(img1: ImageWithPoints, img2: ImageWithPoints):
    pts_match_1, pts_match_2, match_good = ho.matchKeypoints(img1.kp, img1.des, img2.kp, img2.des)
    H, mask = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=0, learning_rate=0.3, method=2)
    H = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with ho.Graph() as graph:
        for i in range(0, 100):
            if i < 50:
                H, mask = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=1, learning_rate=0.15, H=H, mask=mask, method=0, graph=graph, normalization=1)
                image = ho.polyline(ho.project(H, img1.points), img2.image, color=(0, 0, 255))
            else:
                H, mask = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=5, learning_rate=0.3, H=H, mask=mask, method=0, graph=graph, normalization=1)
                image = ho.polyline(ho.project(H, img1.points), img2.image, color=(0, 255, 0))

            image2 = cv2.warpPerspective(img1.image, H, (image.shape[1], image.shape[0]))
            image3 = image2/256*0.25 + image/256*0.75
            cv2.imshow('frame4', image3)

            cv2.imshow('image', image)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break


if __name__ == "__main__":

    frame = cv2.imread("res/graffiti/ref.png")
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    pts = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])
    pts /= 4

    image1 = ImageWithPoints(frame, pts)

    frame = cv2.imread("res/graffiti/14.jpg")
    pts = np.float32([[1335, 452],  [2819, 1061], [2391, 1887], [1137, 1336]])
    pts /= 4
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    image2 = ImageWithPoints(frame, pts)

    frame = cv2.imread("res/graffiti/10.png")
    pts = np.float32([[336, 330],  [443, 182], [616, 317], [452, 439]])
    image3 = ImageWithPoints(frame, pts)

    frame = cv2.imread("res/graffiti/stereo1-100.png")
    pts = np.float32([[296, 117], [463, 84], [479, 205], [323, 250]])
    stereo1 = ImageWithPoints(frame, pts)
    frame = cv2.imread("res/graffiti/stereo2-100.png")
    pts = np.float32([[99, 94], [289, 66], [311, 196], [129, 225]])
    stereo2 = ImageWithPoints(frame, pts)

    frame = cv2.imread("res/graffiti/stereo1-60.png")
    pts = np.float32([[317, 151], [481, 150], [478, 269], [318, 262]])
    stereo3 = ImageWithPoints(frame, pts)
    frame = cv2.imread("res/graffiti/stereo2-60.png")
    pts = np.float32([[179, 131], [312, 137], [306, 262], [177, 240]])
    stereo4 = ImageWithPoints(frame, pts)

    frame = cv2.imread("res/graffiti/16.jpg")
    pts = np.float32()
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    special1 = ImageWithPoints(frame, pts)
    
    frame = cv2.imread("res/graffiti/12.png")
    pts = np.float32()
    special2 = ImageWithPoints(frame, pts)

    print("Vergleich mit der OpenCV Homographiebestimmung")
    compareWithout("images 1-2", image1, image2)
    compareWithout("images 1-3",image1, image3)
    compareWithout("images 2-3",image2, image3)
    compareWithout("stereo 1",stereo1, stereo2)
    compareWithout("stereo 2",stereo3, stereo4)

    print("Vergleich der Verfahren zur Ausreißerdetektion")
    compareRANSACMethod("images 1-2", image1, image2)
    compareRANSACMethod("images 1-3",image1, image3)
    compareRANSACMethod("images 2-3",image2, image3)
    compareRANSACMethod("stereo 1",stereo1, stereo2)
    compareRANSACMethod("stereo 2",stereo3, stereo4)

    print("Vergleich mit initialer Homographie mit Einheitsmatrix")
    CompareRANSACWithout("images 1-2", image1, image2)
    CompareRANSACWithout("images 1-3",image1, image3)
    CompareRANSACWithout("images 2-3",image2, image3)
    CompareRANSACWithout("stereo 1",stereo1, stereo2)
    CompareRANSACWithout("stereo 2",stereo3, stereo4)


    print("Vergleich unterschiedlicher Lernraten")
    compareLearningRate2("images 1-2", image1, image2)
    compareLearningRate2("images 1-3",image1, image3)
    compareLearningRate2("images 2-3",image2, image3)
    compareLearningRate2("stereo 1",stereo1, stereo2)
    compareLearningRate2("stereo 2",stereo3, stereo4)


    print("Grenzen")
    compareLearningRate("special", special1, special2)

    cv2.destroyAllWindows()
