class ParsedLine(object):

    def __init__(self, line):
        self.path = line.split('.jpg')[0] + '.jpg'

        lineData = line.split('.jpg')[1].strip().split(' ')
        self.elementsCount = int(lineData[0])
        self.rects = []

        for i in range(0, self.elementsCount):
            start = i * 4
            x1 = int(lineData[start + 1])
            y1 = int(lineData[start + 2])
            w = int(lineData[start + 3])
            h = int(lineData[start + 4])
            self.rects.append([(x1, y1), (x1 + w, y1 + h)])

    def getRects(self):
        return self.rects
