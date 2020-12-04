from pathlib import Path
from reader.file_reader.page import Page
from reader.se_utils.mupdf_utils import *
import fitz



def _process_page(page, page_id):
    page = parse_page(page)
    page.index = page_id + 1
    return page


class FileReader(object):
    def __init__(self, path: Union[str, Path] = None, stream=None,verbose = False):
        self.path = path
        self.times = []
        if path is not None:
            doc = fitz.Document(path)
        elif stream is not None:
            doc = fitz.Document(stream=stream, filetype='pdf')
        else:
            doc = None
        self.avatars = []
        if doc:
            self.raw_page = len(doc)
            pages: List[Page] = []
            for page_id_, page_ in enumerate(doc):
                pages.append(_process_page(page_, page_id_))
            self.pages = []
            for i, page in enumerate(pages):
                page.analysis()
                self.pages.append(page)
        else:
            self.raw_page = 0
            self.pages = []

    def to_dict(self):
        return [page.to_dict() for page in self.pages]
