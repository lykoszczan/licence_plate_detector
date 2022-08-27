import re

import cv2
import numpy as np
import pytesseract

from src.nms import non_max_supression
from src.utils import scale_image


def process(img):
    # img = cv2.medianBlur(img, 5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    img_blur = cv2.GaussianBlur(thresh, (5, 5), 2)
    img_canny = cv2.Canny(img_blur, 0, 0)
    return img_canny


def get_contours(img):
    cv2.imshow('processed', process(img))
    cv2.waitKey()
    processed = process(img)
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    r1, r2 = sorted(contours, key=cv2.contourArea)[-3:-1]
    x, y, w, h = cv2.boundingRect(np.r_[r1, r2])

    return img[y: y + h, x:x + w]


def detect_licence_plate_characters(test_license_plate, with_contours=True):
    # test_license_plate = cv2.imread('test_data/blurred_plate.png')
    test_license_plate = scale_image(test_license_plate, 300)
    # return ocr_with_segmentation(test_license_plate)
    return ocr_with_segmentation(test_license_plate, True)

    cv2.imshow('before process', test_license_plate)
    cv2.waitKey()
    # print(with_contours)
    custom_oem_psm_config = r'--oem 1 --psm 6 -l eng'

    if with_contours:
        test_license_plate = get_contours(test_license_plate)
    else:
        test_license_plate = process(test_license_plate)

    cv2.imshow('after process', test_license_plate)
    cv2.waitKey()

    predicted_result = pytesseract.image_to_string(test_license_plate, lang='eng',
                                                   config=custom_oem_psm_config)

    if predicted_result:
        letters = re.findall("[A-Z0-9]+", predicted_result)
        predicted_result = ' '.join(letters)

    return predicted_result


def get_candidates(image, cnts, avg_sizes, candidates, counter, verbose):
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if h >= image.shape[0] - 5:
            continue

        if w / h > 2 or h / w > 5 or area < 100:
            if verbose:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 100, 255), 3)
            continue
        counter += 1

        if verbose:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        if area > 1900:
            avg_sizes.append(h)
            candidates.append([x, y, w, h])

    return [avg_sizes, candidates, counter]


def ocr_with_segmentation(image, verbose=False):
    # blue
    candidate_color = (255, 0, 0)
    # grey
    non_max_color = (192, 192, 192)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = np.zeros(image.shape, dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 100, 130])
    mask = cv2.inRange(hsv, lower, upper)

    # Perform morph close and merge for 3-channel ROI extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    extract = cv2.merge([close, close, close])

    # Find contours, filter using contour area, and extract using Numpy slicing

    avg_sizes = []
    candidates = []
    counter = 0

    cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    [avg_sizes, candidates, counter] = get_candidates(image, cnts, avg_sizes, candidates, counter, verbose)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    [avg_sizes, candidates, counter] = get_candidates(image, cnts, avg_sizes, candidates, counter, verbose)

    if not len(avg_sizes) > 0:
        return ''

    avg_height = np.mean(avg_sizes)
    if verbose:
        print('avg_height', avg_height)
        # print('std_dev', statistics.pstdev(avg_sizes))
        # print('variance', statistics.variance(avg_sizes))
        print(np.sort(avg_sizes))

    nms_candidates = []
    for el in candidates:
        x, y, w, h = el
        nms_candidates.append([y, x, h, w])
        if h < (avg_height - 10):
            continue

        if verbose:
            cv2.rectangle(image, (x, y), (x + w, y + h), candidate_color, 3)
        result[y:y + h, x:x + w] = extract[y:y + h, x:x + w]

    rects = non_max_supression(nms_candidates, 0.01)
    for rect in rects:
        if verbose:
            cv2.rectangle(image, rect[0], rect[1], non_max_color, 3)

    if verbose:
        print('contours count', counter)

    invert = 255 - result
    predicted_result = pytesseract.image_to_string(
        invert, lang='eng', config='--psm 6 --oem 1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    if predicted_result:
        predicted_result = predicted_result.replace(')', 'J')
        letters = re.findall("[A-Z0-9]+", predicted_result)
        predicted_result = ' '.join(letters)

    if verbose:
        cv2.imshow('image', image)
        cv2.imshow('Binary image', thresh)
        cv2.imshow('close', close)
        cv2.imshow('invert', invert)
        cv2.waitKey()

    return predicted_result
