import cv2
import numpy as np


class FlashDetector:
    threshold = 20
    radius = 20
    max_noise = 0.01
    min_peak = 0.03
    debug = False

    prev = None

    means = np.zeros(100)


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

            self.means = np.append([np.mean(cv2.mean(thresh))], self.means)
            self.means = self.means[:100]

            if self.means[0] < self.max_noise and self.means[3] < self.max_noise and (self.means[1] > self.min_peak or (self.means[2] > self.min_peak and self.means[1] >= self.max_noise)):
                return True
                
            if self.debug:
                cv2.imshow('thresh'+self.name, thresh)
        self.prev = frame
        return False