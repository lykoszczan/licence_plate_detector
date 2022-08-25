import cv2
import numpy as np

from src.ocr_detection import ocr_with_segmentation

verbose = True

test_dict = {
    'test_data/plate_ocr_test.png': 'ZS750LM',
    'test_data/1661452014.770696.png': 'FZ5055K',
    'test_data/1661452165.056953.png': 'DJ93282',
    'test_data/1661448258.096648.png': 'FZ2942J',
    'test_data/1661454606.268474.png': 'FZ2942J',
    'test_data/1661455167.149186.png': 'FZ2942J',
    'test_data/1661456134.643931.png': 'FZ2942J',
    'test_data/blurred_plate.png': 'FZ5055K',
}

for key in test_dict:
    img = cv2.imread(key)
    scale_height = 300
    h, w, _ = img.shape
    w_new = int(np.round(w * scale_height / h))
    img = cv2.resize(img, (w_new, scale_height))

    value = ocr_with_segmentation(img, verbose)
    value = value.replace(' ', '')

    if value != test_dict[key]:
        print('OCR: ' + str(value) + ' is different from ' + str(test_dict[key]))
        # raise ValueError('OCR: ' + str(value) + ' is different from ' + str(test_dict[key]))
    else:
        print('value correct! ' + str(value))
