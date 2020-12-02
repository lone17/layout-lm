import os
from lxml import html
from bs4 import BeautifulSoup
import re


def get_text(tree):
    soup = BeautifulSoup(html.tostring(tree))
    text = soup.get_text(' ')
    return text.strip()


def parse_color(element):
    '''
    Hàm để đọc thông tin trong html ra mã color và vị trí của color trong bảng
    :param element:
    :return:
    '''
    color = element.get('style')
    color = re.match('.*rgb\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\).*', color).groups()
    color = tuple(int(s) for s in color)
    label = get_text(element)
    return label, color


class ColorTable:
    def __init__(self, path=None):
        if path is None:
            path = os.path.dirname(__file__)
            path = os.path.join(path, 'new_color.html')
        text = open(path, encoding='utf-8').read()
        tree = html.fromstring(text)
        colors_table = list(tree.find_class("ColorPalette"))[0]
        color_rows = list(colors_table.find_class("row"))
        colors = [row.find_class("cell colored") for row in color_rows]
        colors = [x for row in colors for x in row]
        self.idx2color = {}
        for color in colors:
            idx, color = parse_color(color)
            self.idx2color[idx] = color

        self.color2idx = {color: idx for idx, color in self.idx2color.items()}

    def get_idx(self, color):
        '''
        Hàm xác định vị trí của color trong bảng màu xodo
        :param color: Mã màu ví dụ (1,1,1) (0.5, 0.5, 0.5) là mã màu rgb đã chia cho 255
        :return: Vị trí ví dụ 11 là dòng 1 cột 1 trong bảng màu xodo
        '''
        color = tuple(round(i * 255) for i in color)
        label = self.color2idx.get(color, '00')
        if label == '00':
            print('not found', color)
        return label

    def get_color(self, idx):
        '''
        Hàm xác định mã màu rgb ứng với vị trí trong xodo
        :param idx: vị trí, ví dụ: dòng 2 cột 1 thì idx = 21
        :return: Mã màu rgb tại vị trí dòng 2 cột 1
        '''
        return self.idx2color.get(idx, (200, 200, 200))


colortable = ColorTable()
if __name__ == '__main__':
    color = colortable.get_color('23')
    print(color)
    idx = colortable.get_idx((245 / 255, 208 / 255, 169 / 255))
    print(idx)
    print(colortable.get_color(idx))
