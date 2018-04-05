"""Finds keypoints between two images and calculates a homography.
Must be run from the parent directory, e.g.: .../Mehrkamera/src> python -m examples.example1
"""
import numpy as np
import cv2
import modules.homography as ho

image_test = cv2.imread("res/graffiti/1.png")
image_reference = cv2.imread("res/graffiti/ref.png")
corners = np.float32([[54, 26], [2672, 54], [2679, 1905], [21, 1900]])

# resize the reference image
image_reference = cv2.resize(image_reference, (0, 0), fx=0.25, fy=0.25)
corners /= 4


points_reference, points_test, _ = ho.match(image_reference, image_test)

matrix, _ = ho.findHomography(points_reference, points_test)

corners_projected = ho.project(matrix, corners)

test_with_corners = ho.polyline(corners_projected, image_test)

cv2.imshow('test', test_with_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
