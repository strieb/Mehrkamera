import cv2
import numpy as np
import datetime
import threading
import time
import matplotlib.pyplot as plt
import argparse
import sys

videoID = 0
if sys.argv[1]:
    videoID = sys.argv[1]

print(videoID)
cap = cv2.VideoCapture(videoID)

counter = 0


class FlashDetector:
    threshold = 20
    radius = 20
    max_noise = 0.01
    min_peak = 0.03

    prev = None
    pMean = 0
    ppMean = 0
    pppMean = 0
    mean = 0

    def __init__(self, name):
        self.name = name

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            diff = cv2.subtract(frame, self.prev)
            blur = cv2.GaussianBlur(diff, (self.radius * 2 + 1, self.radius * 2 + 1), 0)
            #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO)
            thresh = np.clip(blur, self.threshold, 255)
            thresh = thresh - self.threshold
            self.mean = np.mean(cv2.mean(thresh))
            if self.mean < self.max_noise and self.pppMean < self.max_noise and (self.pMean > self.min_peak or (self.ppMean > self.min_peak and self.pMean >= self.max_noise)):
                print(self.name + ' ' + str(counter))

            self.pppMean = self.ppMean
            self.ppMean = self.pMean
            self.pMean = self.mean
            cv2.imshow('thresh'+str(self.name), thresh)
        self.prev = frame
        cv2.imshow('frame'+str(self.name), frame)


detect = FlashDetector('frame')

means = np.zeros(100)


fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(means, 'o-')
ax.set_ylim([0, 1])
plt.ion()
plt.show()

while(1):
    counter += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    detect.detect(frame)
    means = np.append(means, [detect.mean])
    means = means[-100:]

    Ln.set_ydata(means)
    Ln.set_xdata(range(len(means)))

    key = cv2.waitKey(5)
    if key == 27:  # exit on ESC
        break
cv2.destroyAllWindows()
