from .textline import merge_textlines
import cv2


class Page(object):
    def __init__(self, width, height, textlines, index=1, urls=None, lines=None, cells=None, image=None, images=None,
                 textboxes=None, verbose=False):
        self.index = index
        self.width = int(width)
        self.height = int(height)
        self.textlines = textlines
        if lines is None:
            lines = []
        self.lines = lines
        if cells is None:
            cells = []
        self.cells = cells
        if urls is None:
            urls = []
        if textboxes is None:
            textboxes = []
        self.textboxes = textboxes
        self.urls = urls
        self.image = image
        if self.image is not None:
            self.width = max(self.width, self.image.shape[1])
            self.height = max(self.height, self.image.shape[0])
        if images is None:
            images = []
        self.images = images
        self.avatars = []
        self.tables = []
        self.tree = []
        self.features = None
        self.unit_size = 5
        self.verbose = verbose

    def analysis(self):
        self.textlines = merge_textlines(self.textlines)


    def to_dict(self):
        data = [t.to_dict() for t in self.textlines]
        return {
            'page_index': self.index,
            'height': self.height,
            'width': self.width,
            'textlines': data,
        }

