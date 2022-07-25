import cv2
import os
from os import listdir
from os.path import isfile, join

# python haar_positive_creator.py "../../wideorejestratory" "test.txt"

# path = r'../../video'
path = r'../../wideorejestratory'
savePath = r'../extracted_images'

def createDir(dirPath):
    try:
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

    except OSError:
        print('Error: Creating directory of ' + dirPath)


createDir(savePath)
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
fileIndex = 0

for vidName in onlyfiles:
    createDir(savePath + '/wideorejestratory')
    # createDir(savePath + '/' + vidName)
    fileIndex += 1
    cam = cv2.VideoCapture(join(path, vidName))
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    framesPerImage = round(fps / 3)
    totalFrames = round(frame_count / framesPerImage)
    currentframe = -1
    picIndex = 0

    while (True):
        # reading from frame
        ret, frame = cam.read()
        currentframe += 1

        if ret:
            if currentframe % framesPerImage == 0:
                print('processing file', fileIndex, 'of', len(onlyfiles), 'frame', picIndex, 'of', totalFrames)
                # if video is still left continue creating images
                name = savePath + '/' + 'wideorejestratory' + '/' + vidName + '_' + str(picIndex) + '.jpg'
                # name = savePath + '/' + vidName + '/' + str(picIndex) + '.jpg'
                # print('Creating... ' + name)

                cv2.imwrite(name, frame)
                picIndex += 1
        else:
            break
    cam.release()

cv2.destroyAllWindows()
