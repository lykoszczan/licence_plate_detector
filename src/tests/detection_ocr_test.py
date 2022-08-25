import cv2
import numpy as np
import pytesseract


def ocr_with_segmentation(image):
    h, w, _ = image.shape
    w_new = int(np.round(w * 300 / h))
    image = cv2.resize(image, (w_new, 300))

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Binary image', thresh)
    # cv2.waitKey(0)

    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # imageSharp = cv2.filter2D(image, -1, kernel)

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
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    avg_sizes = []
    candidates = []
    counter = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if w / h > 2 or h / w > 5 or area < 100:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
            continue
        counter += 1
        print(area, x, y)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        if area > 2500:
            avg_sizes.append(h)
            candidates.append([x, y, w, h])
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
            # result[y:y + h, x:x + w] = extract[y:y + h, x:x + w]

    avg_height = np.mean(avg_sizes)
    print('avg_height', avg_height)
    print(np.sort(avg_sizes))

    for el in candidates:
        x, y, w, h = el
        if h < (avg_height * 0.85):
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)
        result[y:y + h, x:x + w] = extract[y:y + h, x:x + w]

    print('contours count', counter)
    # Invert image and throw into Pytesseract
    invert = 255 - result
    data = pytesseract.image_to_string(
        invert, lang='eng', config='--psm 6 --oem 1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    print(data)
    cv2.imshow('image', image)
    # cv2.imshow('imageSharp', imageSharp)
    cv2.imshow('close', close)
    # cv2.imshow('result', result)
    cv2.imshow('invert', invert)
    cv2.waitKey()

    return data


# ocr_with_segmentation(cv2.imread('test_data/plate_ocr_test.png'))
ocr_with_segmentation(cv2.imread('test_data/1661448266.958315.png'))
# ocr_with_segmentation(cv2.imread('test_data/1661448258.096648.png'))
