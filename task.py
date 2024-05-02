
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

image = cv2.imread("/home/mohamed/mine/uni/net pro/images.jpeg")
cv2.imshow('original image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
color = ('blue', 'green', 'red')
for i, col in enumerate(color):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col, label=col + " channel")

plt.xlim([0, 256])
plt.legend()
plt.title("Color Channel Histograms")

plt.show()
plt.close()
alpha = 1.8
beta = -100
adjusted_image = cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)

# display
cv2.imshow('Adjusted Image', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

color = ('blue', 'green', 'red')
for i, col in enumerate(color):
    hist = cv2.calcHist([adjusted_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col, label=col + " channel")

plt.xlim([0, 256])
plt.legend()
plt.title("Color Channel Histograms")

plt.show()
