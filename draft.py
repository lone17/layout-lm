import os
from datasets import InputExample

def read_examples_from_file(data_dir, mode):
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(box_file_path, encoding="utf-8"
              ) as fb, open(image_file_path, encoding="utf-8") as fi:
        words = []
        boxes = []
        actual_bboxes = []
        file_name = None
        page_size = None
        for bline, iline in zip(fb, fi):
            if bline.startswith("-DOCSTART-") or bline == "" or bline == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels='default',
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
            else:
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
                if len(bsplits) == 1 and bsplits.endswith('\n'):
                    continue
                assert len(bsplits) == 2, f"{bsplits} - {isplits}"
                assert len(isplits) == 4
                assert bsplits[0] == isplits[0]
                words.append(bsplits[0])
                box = bsplits[-1].replace("\n", "")
                box = [int(b) for b in box.split()]
                boxes.append(box)
                actual_bbox = [int(b) for b in isplits[1].split()]
                actual_bboxes.append(actual_bbox)
                page_size = [int(i) for i in isplits[2].split()]
                file_name = isplits[3].strip()
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )

    return examples

if __name__ == '__main__':
    folder  = 'D:\\layout-lm\\data\\train'
    read_examples_from_file(folder, 'train')