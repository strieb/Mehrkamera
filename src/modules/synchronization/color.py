import cv2
import numpy as np
import numpy.ma as ma


class ColorDetector:
    debug = False
    means = np.zeros(100)

    _prev = None
    _ptmp = None

    def __init__(self, fps):
        self.fps = fps

    def detect(self, frame) -> bool:
        ret = False
        frame = np.float32(frame) / 255
        blur = cv2.GaussianBlur(frame, (11, 11), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        hsv[:, :, 2] = np.clip(hsv[:, :, 2]*8-1, 0, 1)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1]*4-1, 0, 1)

        if self.debug:
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = np.uint8(img * 255)
            cv2.imshow('color ', img)

        if self._prev is not None:
            level = hsv[:, :, 1] * self._prev[:, :, 1] * hsv[:, :, 2] * self._prev[:, :, 2]
            mask = ma.masked_less(level, 0.5)
            diff = (np.mod(hsv[:, :, 0] - self._prev[:, :, 0] + 180, 360) - 180)
            negative = ma.masked_outside(diff, -150, -90) * mask
            positive = ma.masked_outside(diff, 90, 150) * mask
            self.means = np.append([(negative.count() - positive.count()) / diff.shape[0] / diff.shape[1]], self.means)
            self.means = self.means[:100]

            if(self.means[1] < - 0.002 and self.means[0] > self.means[1]):
                fps = int(self.fps)
                last2secs = self.means[:fps]
                minLoc = last2secs.argmin()
                maxLoc = last2secs.argmax()
                min = last2secs[minLoc]
                max = last2secs[maxLoc]
                if minLoc == 1 and last2secs[3:].min() > min * 0.5 and max > -min * 0.5 and maxLoc > fps*0.4:
                    ret = True

            if self.debug:
                cv2.imshow('mask ', mask)
                cv2.imshow('diff ', (positive.filled(0) + negative.filled(0)) / 360 + 0.5)

        self._prev = self._ptmp
        self._ptmp = hsv
        return ret
