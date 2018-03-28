import cv2
import matplotlib.pyplot as plt
from modules.synchronization import synchronization

cap = cv2.VideoCapture(0)

detector = synchronization.ColorDetector("test", 30)
detector.debug = True


fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(detector.means, 'o-')
ax.set_ylim([0, 5])
plt.ion()
plt.show()


while(1):
    key = cv2.waitKey(1) & 0xFF

    ret, frame = cap.read()
    if not ret:
        break
    if key == 27:  # exit on ESC
        break
    frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_BayerGR2BGR)

    if detector.detect(frame):
        print("detected")
    Ln.set_ydata(detector.means)
    Ln.set_xdata(range(len(detector.means)))

    cv2.imshow('frame', frame)

cv2.destroyAllWindows()
