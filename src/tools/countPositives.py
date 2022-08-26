import numpy as np

import src.utils as utils
from src.objects.ParsedLine import ParsedLine

outputPath = r'data.txt'
extracted_path = r'/home/mlykowski/PycharmProjects/licence_plate_detector/src/extracted_images/wideorejestratory/'

with open(outputPath, 'r') as f:
    lines = f.readlines()

max_plates = 0
counter = 0
all_h = []
all_w = []
max_area = 0
min_area = np.inf
max_dims = (0, 0)
min_dims = (0, 0)
all_img_size = []
for line in lines:
    obj = ParsedLine(line)
    if obj.elementsCount > max_plates:
        max_plates = obj.elementsCount
    counter += obj.elementsCount

    new_path = obj.path.replace('C:\\Users\\lykos\\Desktop\\py-mgr\\src\\extracted_images\\wideorejestratory\\', '')
    try:
        img = utils.readImage(extracted_path + new_path)
        h, w, _ = img.shape
        all_img_size.append((w, h))
        print(len(all_img_size))
    except Exception as e:
        print(e)

    for rect in obj.rects:
        (x1, y1), (x2, y2) = rect
        h = y2 - y1
        w = x2 - x1
        all_h.append(h)
        all_w.append(w)

        area = h * w
        if area > max_area:
            max_area = area
            max_dims = (w, h)

        if area < min_area:
            min_area = area
            min_dims = (w, h)

print('Total positives probes', counter)
print('Max number of plates', max_plates)
print('Mean h', np.mean(all_h))
print('Mean w', np.mean(all_w))
print('Max dims', max_dims)
print('Min dims', min_dims)
print('Resolutions', list(set(all_img_size)))
