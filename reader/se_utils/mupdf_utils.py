import cv2
from typing import *
import fitz
import numpy as np
from reader.file_reader.textline import TextLine, Span
from reader.file_reader.box import sort_box
from reader.se_utils.annotation import get_annotation, merge_textlines_with_annotation
from reader.file_reader.page import Page

LABELS = ['Exp_masterkey', 'Company_name', 'Duration', 'Position', 'Adress']

def pixmap_to_numpy(pix):
    data = pix.getPNGData()
    image = cv2.imdecode(np.fromstring(data, np.uint8), 1)
    return image


def select_font(fonts):
    font = []

    if 'Bold' in fonts:
        font.append('Bold')
    if 'Italic' in fonts:
        font.append('Italic')
    if 'Symbol' in fonts:
        font.append('Symbol')
    if len(font) == 0:
        font.append('Normal')
    return ','.join(font)


def parse_textline(line):
    x0, y0, x1, y1 = line['bbox']
    spans = []
    for span in line['spans']:
        for span_c in span['chars']:
            c_x0, c_y0, c_x1, c_y1 = list(map(int, span_c['bbox']))
            spans.append(Span(c_x0, c_y0, c_x1, c_y1, span_c['c'], span['font'], size=span['size']))
    return TextLine(x0=x0, y0=y0, x1=x1, y1=y1, spans=sorted(spans, key=lambda x: x.x0))


def parse_url(page):
    urls = []
    for link in page.getLinks():
        rect = link['from'].round()
        urls.append({
            'x0': rect.x0,
            'y0': rect.y0,
            'x1': rect.x1,
            'y1': rect.y1,
            'url': link.get('uri', ''),
            'file': link.get('file', '')
        })
    return urls


def list_of_images(page) -> List[np.ndarray]:
    images = []
    doc = page.parent
    if doc is None:
        return images
    for img in page.getImageList():
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n >= 5:  # this is GRAY or RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        images.append(pixmap_to_numpy(pix))
    return images


def parse_image(page, zoom_x=1, zoom_y=1) -> np.ndarray:
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = page.getPixmap(mat, alpha=0)
    data = pix.getPNGData()
    image = np.fromstring(data, np.uint8)  # pixmap_to_numpy(pix)
    image = cv2.imdecode(image, 1)
    return image


def parse_page(page):
    page_dict = page.getText('rawdict')
    blocks = page_dict['blocks']
    images = []
    textlines = []
    for block in blocks:
        if block['type'] == 0:
            for line in block['lines']:
                textlines.extend(parse_textline(line).split('\s+'))
        elif block['type'] == 1:
            img = cv2.imdecode(np.fromstring(block['image'], np.uint8), 1)
            if img is not None:
                images.append(img)

    textlines = [textline for textline in textlines if textline.text.strip() != '']
    if len(textlines) > 0:
        sorted_textlines = sort_box(textlines)
        textlines = [sorted_textlines[0]]
        for textline in sorted_textlines[1:]:
            if textline.is_same_row(textlines[-1]) and textline.x0 < textlines[-1].x1 and textline.x1 > textlines[
                -1].x0:
                textlines[-1].expand(textline)
            else:
                textlines.append(textline)

    annotations = get_annotation(page, labels=LABELS)
    textlines = merge_textlines_with_annotation(annotations, textlines)
    textlines = [textline for textline in textlines
                 if textline.text.strip() != ''
                 and textline.y0 > 0 and textline.y1 < page_dict['height']
                 and textline.x0 > 0 and textline.x1 < page_dict['width']]
    return Page(textlines=textlines,
                width=page_dict['width'],
                height=page_dict['height'],
                urls=parse_url(page),
                image=parse_image(page),
                images=images
                )
