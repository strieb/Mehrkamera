import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import synchronization 

cap = cv2.VideoCapture(1)

detector =  synchronization.FlashDetector("test")
detector.debug = True


fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(detector.means, 'o-')
ax.set_ylim([0.1,-0.1])
plt.ion()
plt.show()


while(1):
    key = cv2.waitKey(1) & 0xFF

    ret, frame = cap.read()
    if not ret:
        break
    if key == 27:  # exit on ESC
        break

    if detector.detect(frame):
        print("detected")
    Ln.set_ydata(detector.means)
    Ln.set_xdata(range(len(detector.means)))

    cv2.imshow('frame', frame)
    

cv2.destroyAllWindows()
