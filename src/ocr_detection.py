import re

import cv2
import numpy as np
import pytesseract

from src.utils import scale_image


def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    img_blur = cv2.GaussianBlur(thresh, (5, 5), 2)
    img_canny = cv2.Canny(img_blur, 0, 0)
    return img_canny


def get_contours(img):
    cv2.imshow('processed', process(img))
    cv2.waitKey()
    contours, _ = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r1, r2 = sorted(contours, key=cv2.contourArea)[-3:-1]
    x, y, w, h = cv2.boundingRect(np.r_[r1, r2])

    return img[y: y + h, x:x + w]


def detect_licence_plate_characters(test_license_plate, with_contours=True):
    # test_license_plate = cv2.imread('test_data/plate_ocr_test.png')
    test_license_plate = scale_image(test_license_plate, 300)

    custom_oem_psm_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    if with_contours:
        test_license_plate = get_contours(test_license_plate)

    cv2.imshow('processed', test_license_plate)
    cv2.waitKey()

    predicted_result = pytesseract.image_to_string(test_license_plate, lang='eng',
                                                   config=custom_oem_psm_config)

    if predicted_result:
        letters = re.findall("[A-Z0-9]+", predicted_result)
        predicted_result = ' '.join(letters)

    return predicted_result
