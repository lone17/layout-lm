from typing import *


class Box(object):
    def __init__(self, x0=-1, y0=-1, x1=-1, y1=-1, idx=-1, tag=None, type='box'):
        self.x0 = int(x0)
        self.x1 = int(x1)
        self.y0 = int(y0)
        self.y1 = int(y1)
        self.idx = idx
        self.tag = tag
        self.type = type
        self.history = []
        self.top: List[Box] = []
        self.bottom: List[Box] = []
        self.left: List[Box] = []
        self.right: List[Box] = []

    def scale(self, dx=1, dy=1):
        '''scale size box'''
        self.x0 *= dx
        self.x1 *= dx
        self.y0 *= dy
        self.y1 *= dy
        self.history.append({'type': 'scale', 'dx': dx, 'dy': dy})

    def shift(self, dx=0, dy=0):
        '''scale size box'''
        self.x0 += dx
        self.x1 += dx
        self.y0 += dy
        self.y1 += dy
        self.history.append({'type': 'shift', 'dx': dx, 'dy': dy})

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def x_cen(self):
        return int((self.x1 + self.x0) / 2)

    @property
    def y_cen(self):
        return int((self.y1 + self.y0) / 2)

    def is_same_row(self, box):
        if box.y0 < self.y_cen < box.y1 or self.y0 < box.y_cen < self.y1:
            return True
        else:
            return False

    def is_intersection_y(self, box):
        height = min(box.height, self.height)
        if box.y0 + 0.15 * height < self.y1 and self.y0 + 0.15 * height < box.y1:
            return True
        else:
            return False

    def is_same_col(self, box):
        if box.x0 < self.x_cen < box.x1 or self.x0 < box.x_cen < self.x1:
            return True
        else:
            return False

    def is_intersection_x(self, box):
        if box.x0 < self.x1 and self.x0 < box.x1:
            return True
        else:
            return False

    def is_intersection(self, box):
        if self.is_intersection_x(box) and self.is_intersection_y(box):
            return True
        else:
            return False

    def __contains__(self, item):
        if self.x0 < item.x_cen < self.x1 and self.y0 < item.y_cen < self.y1:
            return True
        else:
            return False

    def __str__(self):
        return f"x0: {self.x0:0.2f} y0: {self.y0:0.2f} x1: {self.x1:0.2f} y1: {self.y1:0.2f}"

    @property
    def edges(self):
        edges = []
        if self.width > 1:
            edges.append(Line(x0=self.x0, x1=self.x1, y0=self.y0, y1=self.y0))
            if self.height > 0:
                edges.append(Line(x0=self.x0, x1=self.x1, y0=self.y1, y1=self.y1))
        if self.height > 1:
            edges.append(Line(x0=self.x0, x1=self.x0, y0=self.y0, y1=self.y1))
            if self.width > 0:
                edges.append(Line(x0=self.x1, x1=self.x1, y0=self.y0, y1=self.y1))
        return edges

    @property
    def area(self):
        return self.width * self.height

    def iou(self, box):
        x_min = max(self.x0, box.x0)
        x_max = min(self.x1, box.x1)
        y_min = max(self.y0, box.y0)
        y_max = min(self.y1, box.y1)
        if x_max < x_min or y_max < y_min:
            return 0
        else:
            return (x_max - x_min) * (y_max - y_min)

    def to_dict(self):
        return {'x0': self.x0, 'y0': self.y0, 'x1': self.x1, 'y1': self.y1, 'idx': self.idx, 'tag': self.tag,
                'type': self.type}


class Line(Box):
    @property
    def length(self):
        return max(self.height, self.width)


def group_by_row(boxes: List[Box], is_same_row='is_same_row') -> List[List[Box]]:
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda box: box.y0)
    rows = []
    row = [boxes[0]]
    for box in boxes[1:]:
        if getattr(row[-1], is_same_row)(box):
            row.append(box)
        else:
            rows.append(row)
            row = [box]
    if len(row) > 0:
        rows.append(row)
    rows = [sorted(row, key=lambda x: x.x0) for row in rows]
    return rows


def group_by_col(boxes: List[Box]) -> List[List[Box]]:
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda box: box.x0)
    cols = []
    col = [boxes[0]]
    for box in boxes[1:]:
        if col[-1].is_same_col(box):
            col.append(box)
        else:
            cols.append(col)
            col = [box]
    if len(col) > 0:
        cols.append(col)
    cols = [sorted(col, key=lambda x: x.y0) for col in cols]
    return cols


def sort_box(boxes: List[Box], is_same_row='is_same_row') -> List[Box]:
    rows = group_by_row(boxes, is_same_row=is_same_row)
    boxes = []
    for row in rows:
        cols = group_by_col(row)
        for col in cols:
            if len(col) > 0:
                boxes.extend(sorted(col, key=lambda box: box.y_cen))
    return boxes

