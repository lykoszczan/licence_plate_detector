import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, exists


debug = 1
obj_list = []
obj_count = 0
click_count = 0
x1 = 0
y1 = 0
h = 0
w = 0
key = None
frame = None


def readImage(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(path)
        exit(500)

    return img


# mouse callback

def obj_marker(event, x, y, flags, param):
    global click_count
    global debug
    global obj_list
    global obj_count
    global frameName
    global x1
    global y1
    global w
    global h
    global frame
    global frameResized
    if event == cv2.EVENT_LBUTTONDOWN:
        click_count += 1
        if click_count % 2 == 1:
            x1 = x
            y1 = y
        else:
            orgShape = frame.shape
            ratioH = orgShape[0] / 1080
            ratioW = orgShape[1] / 1920

            w = abs(x1 - x)
            h = abs(y1 - y)
            obj_count += 1
            if x1 > x:
                x1 = x
            if y1 > y:
                y1 = y
            obj_list.append('%d %d %d %d ' % (x1 * ratioW, y1 * ratioH, w * ratioW, h * ratioH))
            if debug > 0:
                print(obj_list)
            cv2.rectangle(frameResized, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
            cv2.imshow(frameName, frameResized)


path = r'C:\Users\lykos\Desktop\py-mgr\src\extracted_images\wideorejestratory'
outputPath = r'C:\Users\lykos\Desktop\py-mgr\src\tools\data.txt'
lastIndexPath = r'C:\Users\lykos\Desktop\py-mgr\src\tools\index.txt'

if debug > 0:
    print('Arguments are ok')
    print('Path is : %s' % path)
    print('Output file is : %s' % outputPath)
    print('Click on edges you want to mark as an object')
    print('Press q to quit')
    print('Press c to cancel markings')
    print('Press n to load next image')
# getting list of jpgs files from
list = [f for f in listdir(path) if isfile(join(path, f))]


if exists(lastIndexPath):
    f = open(lastIndexPath, "r")
    index = int(f.read())
    print(index)
else:
    index = 0

list = list[index:]

frameName = str(0) + '/' + str(len(list))

# creating window for frame and setting mouse callback
cv2.namedWindow(frameName, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(frameName, obj_marker)

# loop to traverse through all the files in given path
for currentIndex, i in enumerate(list):
    indexFile = open(lastIndexPath, "w")
    indexFile.write(str(index))
    indexFile.close()
    index += 1

    frameName = str(0) + '/' + str(len(list))

    file_name = open(outputPath, "a")
    i = join(path, i)
    frame = readImage(i)  # reading file

    frameResized = cv2.resize(frame, (1920, 1080))

    cv2.imshow(frameName, frameResized)  # showing it in frame
    obj_count = 0  # initializing obj_count
    key = cv2.waitKey(0)  # waiting for user key

    while ((key & 0xFF != ord('q')) and (key & 0xFF != ord('n'))):  # wait till key pressed is q or n
        key = cv2.waitKey(0)  # if not, wait for another key press
        if (key & 0xFF == ord('c')):  # if key press is c, cancel previous markings
            obj_count = 0  # initializing obj_count and list
            obj_list = []
            frame = readImage(i)  # read original file
            frameResized = cv2.resize(frame, (1920, 1080))
            cv2.imshow(frameName, frameResized)  # refresh the frame
    if (key & 0xFF == ord('q')):  # if q is pressed
        break  # exit
    elif key & 0xFF == ord('n'):  # if n is pressed
        if (obj_count > 0):  # and obj_count > 0
            str1 = '%s %d ' % (i, obj_count)  # write obj info in file
            file_name.write(str1)
            for j in obj_list:
                file_name.write(j)
            file_name.write('\n')
            obj_count = 0
            obj_list = []
    file_name.close()  # end of the program; close the file

cv2.destroyAllWindows()
