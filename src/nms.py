def non_max_supression(detections, ratio):
    rects = []
    for j, k, h, w in detections:
        rects.append([(k, j), (k + w - 1, j + h - 1)])
    run = True
    while run:
        run = False
        for i in range(0, len(rects)):
            for j in range(0, len(rects)):
                if i == j:
                    continue
                if is_intersect(rects[i], rects[j]) and overlapping_ratio(rects[i], rects[j]) > ratio:
                    rec1 = rects.pop(i)
                    if i < j:
                        rec2 = rects.pop(j - 1)
                    else:
                        rec2 = rects.pop(j)
                    rects.append(calculate_new_rectangle(rec1, rec2, overlapping_ratio(rec1, rec2)))
                    run = True
                    break
            if run:
                break
    return rects


def is_intersect(rec1, rec2):
    if (rec1[0][0] < rec2[0][0]) or (rec1[0][1] < rec2[0][1]):
        if (rec1[1][0] > rec2[0][0]) or (rec1[1][1] > rec2[0][1]):
            return True
    elif (rec1[0][0] > rec2[0][0]) or (rec1[0][1] > rec2[0][1]):
        if (rec1[0][0] < rec2[1][0]) or (rec1[0][1] < rec2[1][1]):
            return True
    return False


def overlapping_ratio(rec1, rec2):
    rec1_area = (rec1[1][0] - rec1[0][0]) * (rec1[1][1] - rec1[0][1])
    rec2_area = (rec1[1][0] - rec1[0][0]) * (rec1[1][1] - rec1[0][1])

    xx = max(rec1[0][0], rec2[0][0])
    yy = max(rec1[0][1], rec2[0][1])
    aa = min(rec1[1][0], rec2[1][0])
    bb = min(rec1[1][1], rec2[1][1])

    width = max(0, aa - xx)
    height = max(0, bb - yy)

    intersection_area = width * height

    union_area = rec1_area + rec2_area - intersection_area

    return intersection_area / union_area


def calculate_new_rectangle(rec1, rec2, ratio):
    new_rec = []
    if ratio > 0.3:
        new_rec.append((int((rec1[0][0] + rec2[0][0]) / 2), int((rec1[0][1] + rec2[0][1]) / 2)))
        new_rec.append((int((rec1[1][0] + rec2[1][0]) / 2), int((rec1[1][1] + rec2[1][1]) / 2)))
    else:
        xx = min(rec1[0][0], rec2[0][0])
        yy = min(rec1[0][1], rec2[0][1])
        aa = max(rec1[1][0], rec2[1][0])
        bb = max(rec1[1][1], rec2[1][1])
        new_rec.append((xx, yy))
        new_rec.append((aa, bb))
    return new_rec
