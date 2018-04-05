""" This example combines two images into a larger image using a homography projection.
"""
import cv2
import modules.homography as ho
import numpy as np

image_1 = cv2.imread("res/graffiti/9.png")
image_2 = cv2.imread("res/graffiti/10.png")


pts_1, pts_2, all_keypoints = ho.match(image_1, image_2)

H, mask = ho.findHomography(pts_1, pts_2)

image_keypoints = ho.drawMatches(all_keypoints, image_1, image_2, mask)
cv2.imshow('keypoints', image_keypoints)

image_1_warp = cv2.warpPerspective(image_1, H, (1000, 800))
image_2_extended = np.zeros((800, 1000, 3), np.uint8)
image_2_extended[:480, :640, :] = image_2
image_blend = image_1_warp/256 * 0.5 + image_2_extended/256 * 0.5
cv2.imshow('blend', image_blend)

cv2.waitKey(0)
cv2.destroyAllWindows()
