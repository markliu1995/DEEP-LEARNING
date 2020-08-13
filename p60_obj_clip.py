import numpy as np


def obj_clip(img, foreground, border=None):
    """

    :param img: np.ndarray
    :param foreground:
    :return:  [[[x1, y1], [x2, y2], ....],  [[x1, y1], [x2, y2], ...], ....]

    """
    result = []
    height, width = np.shape(img)
    visited = set()
    for h in range(height):
        for w in range(width):
            if img[h, w] == foreground and not (h, w) in visited:
                obj = visit(img, height, width, h, w, visited, foreground, border)
                result.append(obj)
    return result


def visit(img, height, width, h, w, visited, foreground, border):
    visited.add((h, w))
    result = [(h, w)]

    if w > 0 and not (h, w-1) in visited:
        if img[h, w-1] == foreground:
            result += visit(img, height, width, h, w-1, visited, foreground, border)
        elif border is not None and img[h, w-1] == border:
            result.append((h, w-1))
    if w < width - 1 and not (h, w+1) in visited:
        if img[h, w+1] == foreground:
            result += visit(img, height, width, h, w+1, visited, foreground, border)
        elif border is not None and img[h, w+1] == border:
            result.append((h, w+1))

    if h > 0 and not (h-1, w) in visited:
        if img[h-1, w] == foreground:
            result += visit(img, height, width, h-1, w, visited, foreground, border)
        elif border is not None and img[h-1, w] == border:
            result.append((h-1, w))
    if h < height - 1 and not (h+1, w) in visited:
        if img[h+1, w] == foreground:
            result += visit(img, height, width, h+1, w, visited, foreground, border)
        elif border is not None and img[h+1, w] == border:
            result.append((h+1, w))
    return result


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(20000)
    import cv2
    img = np.zeros([400, 400])
    cv2.rectangle(img, (10,10), (150, 150), 1.0, 1)
    cv2.circle(img, (270, 270), 70, 1.0, 10)
    cv2.line(img, (100, 10), (100, 150), 0.5, 10)
    cv2.putText(img, 'ABC', (200, 200), cv2.FONT_HERSHEY_PLAIN, 2.0, 1.0, 2)

    cv2.imshow('aaa', img)
    cv2.waitKey()
    for obj in obj_clip(img, 1.0, 0.5):
        clip = np.zeros([400, 400])
        for h, w in obj:
            clip[h, w] = 1.0
        cv2.imshow('aaa', clip)
        cv2.waitKey()
