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
    
    frame1 = cv2.imread("images/graffiti/4.png")
    frame2 = cv2.imread("images/graffiti/10.png")
    pts1, pts2, kp1, kp2, good = ho.match(frame1,frame2)

    N = ho.findNormalizationMatrixWithSize(frame1.shape)
    src = ho.project(N,pts1)
    dst = ho.project(N,pts2)

    pts = np.float32([[81, 69], [418, 63], [420, 277], [113, 315]]).reshape(-1, 1, 2)

    H_cv2, mask_cv2 = cv2.findHomography(src, dst, cv2.RANSAC, 0.008)
    mask_cv2 = mask_cv2.ravel()
    mask_cv2 = mask_cv2 > 0.5

    with ho.Graph() as graph:
        src_pts = src
        dst_pts = dst
        
        H, mask = ho.ransac2(src, dst, 0.015)
        print(mask_cv2)
        print(mask)

        #mask = mask_cv2
        graph.assign(H)
      
      #  print(mask.sum())

        graph.train(src_pts, dst_pts,mask=mask, training_epochs=10, learning_rate=0.3)
        H = graph.currentMatrix()

        err = ho.distance_error(H,src_pts, dst_pts)
        mask = err < 0.008
        print(mask.sum())

        graph.train(src_pts, dst_pts,mask=mask, training_epochs=30, learning_rate=0.1)
        H = graph.currentMatrix()
    

    H = np.matmul(np.linalg.inv(N),np.matmul(H,N))
    H /= H[2,2]


    H_cv2 = np.matmul(np.linalg.inv(N),np.matmul(H_cv2,N))
    H_cv2 /= H_cv2[2,2]
    
    print(H)
    print(H_cv2)


    print("mask")
    print(np.sum(mask))
    print("mask_CV2")
    print(np.sum(mask_cv2))
    print("TF")
    print(ho.goldStandardError(H, pts1, pts2, mask).mean())
    print(ho.goldStandardError(H, pts1, pts2, mask_cv2).mean())
    print(ho.distance_error(H, pts1, pts2, mask).mean())
    print(ho.distance_error(H, pts1, pts2, mask_cv2).mean())
    print("CV2")
    print(ho.goldStandardError(H_cv2, pts1, pts2, mask).mean())
    print(ho.goldStandardError(H_cv2, pts1, pts2, mask_cv2).mean())
    print(ho.distance_error(H_cv2, pts1, pts2, mask).mean())
    print(ho.distance_error(H_cv2, pts1, pts2, mask_cv2).mean())

    mask_int = (mask * 1).tolist()
    while True:
        frame3 = ho.visualize(kp1, kp2, good, frame1,frame2,H,H_cv2,mask_int, pts)
        cv2.imshow("test", frame3)   
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    