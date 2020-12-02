from reader.file_reader.textline import Span, TextLine, TextBox
from reader.xodo.color import colortable
from typing import *


def boxhit(item, box, item_text):
    item_x0, item_y0, item_x1, item_y1 = item
    x0, y0, x1, y1 = box
    assert item_x0 <= item_x1 and item_y0 <= item_y1
    assert x0 <= x1 and y0 <= y1

    # does most of the item area overlap the box?
    # http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    x_overlap = max(0, min(item_x1, x1) - max(item_x0, x0))
    y_overlap = max(0, min(item_y1, y1) - max(item_y0, y0))
    overlap_area = x_overlap * y_overlap
    item_area = (item_x1 - item_x0) * (item_y1 - item_y0)
    assert overlap_area <= item_area

    if item_area == 0:
        return False
    else:
        return overlap_area >= 0.5 * item_area


def get_annot_character(vertices, chars, label=''):
    components = []
    if vertices is None:
        return []
    while len(vertices) > 0:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = vertices[:4]
        vertices = vertices[4:]
        xvals = [x0, x1, x2, x3]
        yvals = [y0, y1, y2, y3]
        x0, y0, x1, y1 = min(xvals), min(yvals), max(xvals), max(yvals)
        inside = []
        for c in chars:
            if boxhit((c['x0'], c['y0'], c['x1'], c['y1']), (x0, y0, x1, y1), c['char']):
                c['label'] = label
                inside.append(c)
        inside = sorted(inside, key=lambda c: c['x0'])
        spans = [Span(c['x0'], c['y0'], c['x1'], c['y1'], c['char'], c['font'], c['size']) for c in inside]
        components.append(TextLine(x0=x0, y0=y0, x1=x1, y1=y1, spans=spans, label=label))
    return components


def get_annotation(page, labels=None):
    chars = []
    rawDict = page.getText('rawdict')
    for k in rawDict['blocks']:
        if k['type'] == 0:
            for line in k['lines']:
                for span in line['spans']:
                    for span_c in span['chars']:
                        x0, y0, x1, y1 = list(map(int, span_c['bbox']))
                        # pad = (y1-y0)//2
                        # print(span_c)
                        chars.append({
                            'char': span_c['c'],
                            'x0': x0,
                            'y0': y0,
                            'x1': x1,
                            'y1': y1,
                            'font': span['font'],
                            'size': span['size']
                        })
        # else:
        #     print(k)
        #     1/0
    annot = page.firstAnnot
    words = page.getTextWords()
    annotations = []
    tag = 0
    while annot:
        # print_annot(annot)
        x0, y0, x1, y1 = annot.rect
        # label = get_label(annot.colors['stroke'])
        # if labels is not None and label not in labels:
        #     annot = annot.next
        #     continue
        # text = annot.info['content']
        # print("label: ", text)
        label = colortable.get_idx(annot.colors['stroke'])
        # print(label)
        # print(labels)
        if labels is not None and label not in labels:
            annot = annot.next
            continue
        # if labels is not None:
        #     label = labels[label]

        annots = get_annot_character(annot.vertices, chars, label)
        annotations.append(TextBox(textlines=annots, tag=tag))
        annot = annot.next
        tag += 1

    return annotations


def merge_textlines_with_annotation(annotations: List[TextBox], textlines: List[TextLine]):
    result = []
    for textline in textlines:
        anno_interact = []
        for annotation in annotations:
            for anno in annotation:
                if anno.is_same_row(textline) and anno.iou(textline) > 0.5 * textline.height ** 2:
                    anno.tag = annotation.tag
                    anno_interact.append(anno)
        if len(anno_interact) > 0:
            anno_interact = sorted(anno_interact, key=lambda x: x.x0)
            for anno in anno_interact:
                x_end = min(anno.x1, textline.x1)
                for c in textline.spans:
                    if anno.x0 < c.x_cen < x_end:
                        c.label = anno.label
            result.append(textline)
        else:
            result.append(textline)

    return result
