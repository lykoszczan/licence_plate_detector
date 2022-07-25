import cv2
import utils
from objects.ParsedLine import ParsedLine
import numpy
import time

# for i in range(10):
#     time.sleep(1)
#     print(numpy.random.random())
# exit(1)

lines = utils.readDataFile()

lastLine = lines[-1]
obj = ParsedLine(lastLine)

im = utils.readImage(obj.path)

for i in range(0, obj.elementsCount):
    cv2.rectangle(im, obj.rects[i][0], obj.rects[i][1], (0, 0, 255), 1)

imResized = cv2.resize(im, (1920, 1080))
cv2.imshow(obj.path, imResized)
cv2.waitKey(0)