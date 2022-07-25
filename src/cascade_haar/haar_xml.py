import cv2
import numpy as np
import re
from PIL import Image, ImageFilter
import pytesseract
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def character_recognition(found_plate, plate_box, img):
    # plt.imshow(found_plate)
    # plt.show()

    config = ("-l eng --oem 1 --psm 3")
    testText = pytesseract.image_to_string(found_plate, config=config)
    x = re.findall("[A-Z0-9]+", testText)
    print('regex', x)

    plate_text = ' '.join(x)

    # plate_text = 'ZS750LM'
    # Get bounding box position of plate
    x, y, w, h = plate_box

    # Print Recognize text on image
    copy_img = np.copy(img)
    # Specify font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # draw text
    cv2.putText(copy_img, plate_text, (x, y + h + 10), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("License plate number recognition", copy_img)
    print('Plate text:', plate_text)


def main():
    # Measure time
    e1 = cv2.getTickCount()

    # Load cascade model
    cascade = cv2.CascadeClassifier("data.xml")

    # Load image
    img = cv2.imread("../extracted_images/30.jpg")
    # img = cv2.imread("../test_data/car.png")

    # Resize image
    width = 600
    height = int(600 * img.shape[0] / img.shape[1])
    # img = cv2.resize(img, (width, height))

    # Convert to gray image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalize brightness and increase the contrast
    grayImg = cv2.equalizeHist(grayImg)

    # detect license plate
    box_plates = cascade.detectMultiScale(grayImg, scaleFactor=1.05, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw bounding box on detected plate
    for (x, y, w, h) in box_plates:
        # Draw bounding box
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Plate roi
        plate_roi = np.copy(img[y:y + h, x:x + w])
        # cv2.imshow("Plate ROI", plate_roi)
        # OCR
        # character_recognition(plate_roi, (x, y, w, h), img)

    cv2.imshow("Plate detection", img)

    # End time
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print('Time: %.2f(s)' % (time))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
