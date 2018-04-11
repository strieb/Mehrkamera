"""Visualizes the optimization process.
"""
import numpy as np
import cv2
import modules.homography as ho

image_test = cv2.imread("res/graffiti/11.png")
image_reference = cv2.imread("res/graffiti/ref.png")
corners = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])

# resize the reference image


image_reference = cv2.resize(image_reference, (0, 0), fx=0.25, fy=0.25)
corners /= 4


pts_match_1, pts_match_2, _ = ho.match(image_reference, image_test)

H = np.identity(3)
_, mask = ho.findHomography(pts_match_1, pts_match_2)

# wait
cv2.imshow('frame', image_test)
k = cv2.waitKey(1000) & 0xFF

with ho.Graph() as graph:
    iter = 0
    for i in range(0, 140):
        if i < 50:
            iter += 1
            # 50 times 1 epoch
            H, mask = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=1, learning_rate=0.3, H=H, mask=mask, method=0, graph=graph, normalization=1)
            error = ho.distanceError(H, pts_match_1, pts_match_2, mask).mean()
            image = ho.polyline(ho.project(H, corners), image_test, color=(0, 0, 255)) #red
            image = cv2.putText(image,"iteration "+str(iter)+",  error: "+str(error),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            
        else:
            iter += 5
            # 90 times 5 epochs
            H, mask = ho.findHomography(pts_match_1, pts_match_2, 2, epochs=5, learning_rate=0.3, H=H, mask=mask, method=0, graph=graph, normalization=1)
            error = ho.distanceError(H, pts_match_1, pts_match_2, mask).mean()
            image = ho.polyline(ho.project(H, corners), image_test, color=(0, 255, 0)) #green
            image = cv2.putText(image,"iteration "+str(iter)+", error: "+str(error),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)

        image2 = cv2.warpPerspective(image_reference, H, (image_test.shape[1], image_test.shape[0]))
        image3 = image2/256*0.5 + image/256*0.5
        cv2.imshow('frame', image3)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
