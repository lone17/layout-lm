from .box import Box
import re
from .box import sort_box
import numpy as np
from typing import *

new_line_symbol = ['-', '+', '', '', '', '', '', '•', 'ü', '', '●']
merge_to_right_symbol = ['', '', '', '', '', '•', '', '●', ]


def get_bullet(text):
    text = text.strip().replace(' ', '')
    bullet = '•●*'
    if len(text) > 0:
        if text[0] in bullet and (len(text) == 1 or text[1] != text[0]):
            return 'bullet'
        elif text[0] in '-+':
            return text[0]
        else:
            return None
    else:
        return None


class Span(Box):
    def __init__(self, x0, y0, x1, y1, text, font='Normal', size=-1, label=None):
        Box.__init__(self, x0, y0, x1, y1)
        self.text = text
        font_ = []
        if 'Bold' in font or 'Impact' in font:
            font_.append('Bold')
        if 'Italic' in font:
            font_.append('Italic')
        if 'Symbol' in font or 'Wingding' in font:
            font_ = ['Symbol']
        if len(font_) == 0:
            font_.append('Normal')
            # font_.append(font)
        self.font = ','.join(font_)
        self.size = size
        self.label = label

    def to_dict(self):
        d = Box.to_dict(self)
        d['font'] = self.font
        d['size'] = self.size
        d['text'] = self.text
        d['label'] = self.label
        return d

    def __repr__(self):
        return '{0:0.2f} {1:0.2f} {2:0.2f} {3}'.format(self.x0, self.x1, self.height, self.text)


class Word(Box):
    def __init__(self, chars, start_position=-1):
        Box.__init__(self)
        self.chars = chars
        self.x0 = min([c.x0 for c in chars])
        self.x1 = max([c.x1 for c in chars])
        self.y0 = min([c.y0 for c in chars])
        self.y1 = min([c.y1 for c in chars])
        self.text = ''.join([c.text for c in sorted(chars, key=lambda x: x.x0)])
        self.start_position = start_position

    def to_dict(self):
        d = Box.to_dict(self)
        d['text'] = self.text
        d['start_position'] = self.start_position
        return d

    def __repr__(self):
        return '{0:0.2f} {1:0.2f} {2:0.2f} {3}'.format(self.x0, self.x1, self.height, self.text)


class TextLine(Box):
    def __init__(self, x0, y0, x1, y1, spans, label=None, tag=None):
        Box.__init__(self, x0=x0, y0=y0, x1=x1, y1=y1, tag=tag, type='textline')
        if len(spans) == 0:
            self.spans = spans
        else:
            skip_ids = set()
            height = np.mean([s.height for s in spans])
            for i, span in enumerate(spans):
                if span.text.strip() == '':
                    if i == 0 or i == len(spans) - 1:
                        skip_ids.add(i)
                    else:
                        if spans[i].x1 - spans[i].x0 < height / 7:
                            skip_ids.add(i)
            self.spans = [span for i, span in enumerate(spans) if i not in skip_ids]
        for i, c in enumerate(self.spans):
            if c.text.strip() != '':
                self.x0 = c.x0
                self.spans = self.spans[i:]
                break
        for i, c in enumerate(reversed(self.spans)):
            if c.text.strip() != '':
                self.x1 = c.x1
                self.spans = self.spans[:len(self.spans) - i]
                break
        self.answers = []
        self.text = ''
        prev = None
        start = 0
        for span in self.spans:
            if span.label != prev:
                if prev != None:
                    self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})
                if span.label != None:
                    start = len(self.text)
            self.text += span.text
            prev = span.label
        if prev != None:
            self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})

        self.label = label
        if len(self.spans) == 0:
            self.font = ['Normal']
            self.start_with_bullet = False
        else:
            self.font = [span.font for span in self.spans]
            if 'Symbol' in self.spans[0].font:
                self.start_with_bullet = True
            elif get_bullet(self.text) == 'bullet':
                self.start_with_bullet = True
            else:
                self.start_with_bullet = False
        self.words = []

    @property
    def size(self) -> int:
        if len(self.spans) == 0:
            return 0
        else:
            return max(span.size for span in self.spans)

    def __str__(self):
        return "{0} label: {1} text: {2}".format(Box.__str__(self), self.label, self.text)

    def to_dict(self):
        self.words = self.split(return_word=True)  # split into words
        d = Box.to_dict(self)
        self.answers = []
        self.text = ''
        prev = None
        start = 0
        for span in self.spans:
            if span.label != prev:
                if prev != None:
                    self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})
                if span.label != None:
                    start = len(self.text)
            self.text += span.text
            prev = span.label
        words = [w.to_dict() for w in self.words]  # add words
        if prev != None:
            self.answers.append({'start_pos': start, 'text': self.text[start:], 'label': prev})
        d.update({
            'text': self.text, 'font': ','.join(set(self.font)),
            'size': self.size, 'label': self.label,
            'words': words,
            'answers': self.answers
        })
        return d

    def split(self, sep=None, min_distance=0.5, return_word=False):
        words = []
        chars = self.spans
        text = ''.join(span.text for span in chars)
        if sep == None:
            sep = '\s+'
            min_distance = 0.1
        end = 0
        textlines = []
        for span in re.finditer(sep, text):
            if chars[min(span.end(), len(chars) - 1)].x0 - chars[
                max(span.start() - 1, 0)].x1 < min_distance * self.height:
                continue
            if span.start() > end:
                textlines.append(
                    TextLine(chars[end].x0, self.y0, chars[span.start() - 1].x1, self.y1, chars[end:span.start()],
                             self.label))
                words.append(Word(chars[end:span.start()], end))
            end = span.end()
        if end < len(chars):
            textlines.append(
                TextLine(chars[end].x0, self.y0, chars[len(chars) - 1].x1, self.y1, chars[end:],
                         self.label))
            words.append(Word(chars[end:], end))
        if return_word:
            return words
        else:
            return textlines

    def expand(self, textline):
        if self.is_same_row(textline):
            spans = self.spans + textline.spans
            sorted_spans = sorted(spans, key=lambda b: b.x0)
            spans = [sorted_spans[0]]
            for span in sorted_spans[1:]:
                if span in spans[-1]:
                    if 'Normal' in spans[-1].font:
                        spans[-1].font = 'Duplicate'
                    else:
                        if 'Duplicate' not in spans[-1].font:
                            spans[-1].font += ',Duplicate'
                else:
                    spans.append(span)
            self.__init__(sorted_spans[0].x0, min(self.y0, textline.y0), sorted_spans[-1].x1, max(self.y1, textline.y1),
                          spans, self.label, self.tag)


def can_merge_to_right(textline1: TextLine, textline2: TextLine):
    if abs(textline1.y_cen - textline2.y_cen) > 0.1 * (textline1.height + textline2.height):
        return False
    if textline1.font[-1] == 'Symbol':
        return True
    elif textline2.start_with_bullet:
        return False
    elif len(textline1.text.strip()) > 0 and textline1.text.strip()[-1] in merge_to_right_symbol:
        return True
    elif textline2.text.strip()[0] in [':']:
        return True
    elif textline2.x0 - textline2.height / 2 < textline1.x1:
        return True
    elif textline2.x0 - textline2.height / 1.9 < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.1 * (
            textline1.height + textline2.height):
        return True
    elif textline2.x0 - textline2.height / 1.7 < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.01 * (
            textline1.height + textline2.height):
        return True
    elif textline2.x0 - textline2.height < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.01 * (
            textline1.height + textline2.height) and len(textline2.text.split()) < 2:
        return True
    elif textline2.x0 - textline2.height * 2 < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.1 * (
            textline1.height + textline2.height) and textline1.text.strip()[-1] in new_line_symbol:
        return True
    elif textline2.x0 - textline2.height * 2.5 < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.1 * (
            textline1.height + textline2.height) and (
            len(textline1.text.split()) < 2 or len(textline2.text.split()) < 2):
        return True
    elif textline2.x0 - textline2.height * 3.5 < textline1.x1 and abs(textline1.y_cen - textline2.y_cen) < 0.1 * (
            textline1.height + textline2.height) and (
            len(textline1.text.strip()) < 2 or len(textline2.text.strip()) < 2):
        return True
    elif re.fullmatch('(?:\d+|[XIV]+)\s*[\.\:\.\/\,\)]{0,1}.*', textline1.text.strip()) and len(
            textline1.text.strip()) < 6:
        return True
    else:
        return False


def merge_textlines(textlines):
    if len(textlines) == 0:
        return []
    sorted_textlines = sort_box(textlines)
    textlines = [sorted_textlines[0]]
    for textline in sorted_textlines[1:]:
        if textline.is_same_row(textlines[-1]) and can_merge_to_right(textlines[-1], textline):
            textlines[-1] = combie_textline(textlines[-1], textline)
        else:
            textlines.append(textline)
    return textlines


def combie_textline(self, textline):
    if self.is_same_row(textline):
        spans = self.spans + textline.spans
        sorted_spans = sorted(spans, key=lambda b: b.x0)
        spans = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            if span in spans[-1]:
                if 'Normal' in spans[-1].font:
                    spans[-1].font = 'Duplicate'
                else:
                    if 'Duplicate' not in spans[-1].font:
                        spans[-1].font += ',Duplicate'
            else:
                spans.append(span)
        return TextLine(sorted_spans[0].x0, min(self.y0, textline.y0), sorted_spans[-1].x1, max(self.y1, textline.y1),
                        spans, self.label, self.tag)
    else:
        raise AttributeError("textline not same row")


class BoxContainer(Box):
    def __init__(self, boxes: List[Box] = None, type='textbox'):
        Box.__init__(self, type=type)
        if boxes is None:
            boxes = []
        self.boxes = boxes
        self.update_position()
        # self.is_title = is_title

    def __iter__(self):
        for box in self.boxes:
            yield box

    def append(self, box):
        self.boxes.append(box)
        if self.x0 < 0:
            self.x0 = box.x0
        else:
            self.x0 = min(self.x0, box.x0)

        if self.x1 < 0:
            self.x1 = box.x1
        else:
            self.x1 = max(self.x1, box.x1)

        if self.y0 < 0:
            self.y0 = box.y0
        else:
            self.y0 = min(self.y0, box.y0)

        if self.y1 < 0:
            self.y1 = box.y1
        else:
            self.y1 = max(self.y1, box.y1)

    def extend(self, boxes):
        self.boxes.extend(boxes)
        self.update_position()

    def push(self, boxes):
        self.boxes = boxes + self.boxes
        self.update_position()

    def update_position(self):
        if len(self) > 0:
            self.x0 = min(map(lambda t: t.x0, self.boxes))
            self.x1 = max(map(lambda t: t.x1, self.boxes))
            self.y0 = min(map(lambda t: t.y0, self.boxes))
            self.y1 = max(map(lambda t: t.y1, self.boxes))

    def __len__(self):
        return len(self.boxes)

    def to_dict(self):
        d = Box.to_dict(self)
        titles = []
        childs = []
        for b in self.boxes:
            if hasattr(b, 'is_title') and b.is_title and len(titles) == 0:
                titles.append(b)
            else:
                childs.append(b)
        if len(titles) > 0:
            title = titles[0].to_dict()
            d['text'] = title['text']
            d['answers'] = title['answers']
            d['is_title'] = True
        else:
            d['text'] = None
            d['answers'] = []
            d['is_title'] = False
        d['children'] = [b.to_dict() for b in childs]
        return d


class TextBox(BoxContainer):
    def __init__(self, textlines: List[TextLine], tag=None):
        BoxContainer.__init__(self, boxes=textlines, type='textbox')
        self.tag = tag

    def __iter__(self):
        for textline in self.textlines:
            yield textline

    def append(self, textline):
        BoxContainer.append(self, textline)

    def extend(self, textlines):
        BoxContainer.append(self, textlines)

    @property
    def textlines(self) -> List[TextLine]:
        return self.boxes
