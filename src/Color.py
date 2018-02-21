import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)

prev = None

means = np.zeros(100)

fig = plt.figure()
ax = fig.add_subplot(111)
Ln, = ax.plot(means, 'o-')
ax.set_ylim([-1, 1])
plt.ion()
plt.show()


while(1):
    key = cv2.waitKey(20) & 0xFF

    ret, frame = cap.read()
    if not ret:
        break
    if key == 27:  # exit on ESC
        break


    frame = np.float32(frame) / 255
    blur = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #print(np.max(hsv[:,:,2]))
    hsv[:,:,2] = np.clip(hsv[:,:,2]*8-1,0,1)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*3-1,0,1)
    #hsv[:,:,0] = i
    #hsv = np.float32(hsv)
    #test -= 128
    #hsv = np.clip(hsv,0,1)

    #img = np.uint8(hsv)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
    img = np.uint8(img * 255)
    cv2.imshow('test', img)
    
    if prev is not None: 
        diff = (np.mod(hsv[:,:,0]-prev[:,:,0] + 180, 360) - 180) * (hsv[:,:,1] + prev[:,:,1]) * (hsv[:,:,2] + prev[:,:,2]) / 4
        #diff = np.clip(diff-10,0,360)
        means = np.append(means, [np.mean(diff)])
        means = means[-100:]

        Ln.set_ydata(means)
        Ln.set_xdata(range(len(means)))

        cv2.imshow('diff', diff / 360 + 0.5)

    prev = hsv


    #frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame', frame)
    

cv2.destroyAllWindows()
