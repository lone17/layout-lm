from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import numpy as np
import os
import json
from datasets import read_examples_from_file
# from transformers import AutoTokenizer


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def get_color_tag(tag):
    if tag == "B":
        color = 'red'
    elif tag == "E":
        color = 'blue'
    elif tag == "S":
        color = 'magenta'
    elif tag == 'I':
        color = 'green'
    else:
        color = 'gray'
    return color


def plot_single_example(lines, export_path, scale_factor=5):
    example, preds = lines
    image = Image.fromarray(np.ones((scale_factor * 1000, scale_factor * 1000, 3), dtype='uint8') * 255)
    draw = ImageDraw.Draw(image)

    if len(example.words) != len(preds):
        print('Len not matched {} vs {}'.format(len(example.words), len(preds)))
        print('Stop')
        exit(1)

    for text, label, box, pred in zip(example.words, example.labels, example.boxes, preds):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = [k * scale_factor for k in [x1, y1, x2, y2]]

        draw.text((x1 + 40, y1), text, fill='black', font=font)
        draw_rectangle(draw, ((x1, y1), (x2, y2)), get_color_tag(label[0]), width=6)
        draw_rectangle(draw, ((x1 + 50, y1 + 50), (x2 - 50, y2 - 50)), get_color_tag(pred[0]), width=10)

    image.save(export_path)


def plot_single(lines, export_path, scale_factor=5):
    lines, lines_with_label = lines
    image = Image.fromarray(np.ones((scale_factor * 1000, scale_factor * 1000, 3), dtype='uint8') * 255)
    draw = ImageDraw.Draw(image)
    for line, line_label in zip(lines, lines_with_label):
        item = line.split()
        # print(item)
        if len(item) < 5:
            continue
        text = ' '.join(item[:-4])
        # tokens = tokenizer.tokenize(text)
        x1, y1, x2, y2 = int(item[-4]), int(item[-3]), int(item[-2]), int(item[-1])
        x1, y1, x2, y2 = [k * scale_factor for k in [x1, y1, x2, y2]]
        label = line_label.split()[-1]
        color = get_color_tag(label[0])

        draw.text((x1, y1), text, fill='black', font=font)
        draw_rectangle(draw, ((x1, y1), (x2, y2)), color, width=4)

    image.save(export_path)


def read_file_txt(input_path):
    with open(input_path, 'r') as fi:
        lines = fi.readlines()
        size = len(lines)
        idx_list = [idx + 1 for idx, val in
                    enumerate(lines) if val == '\n']
        res = [lines[i: j] for i, j in
               zip([0] + idx_list, idx_list +
                   ([size] if idx_list[-1] != size else []))]
    return res


def get_latest_output(pred_dict):
    pred_dict = {int(key.split('-')[1]): value for key, value in pred_dict.items()}
    max_idx = max(pred_dict.keys())
    return pred_dict[max_idx]


mode = 'val'
input_data_box = 'data_toshiba/{}_box.txt'.format(mode)
input_data_label = 'data_toshiba/{}.txt'.format(mode)
pred_output = 'eval_preds.json'

output_path = 'visualize'
font = ImageFont.truetype(
    'Dengb.ttf',
    size=20,
    encoding='utf-8-sig')

# tokenizer = AutoTokenizer.from_pretrained(
#     'bert-base-uncased', do_lower_case=True
# )

if not os.path.isdir(output_path):
    os.mkdir(output_path)

list_lines = read_file_txt(input_data_box)
list_lines_with_label = read_file_txt(input_data_label)

examples = read_examples_from_file(data_dir='data_toshiba',
                                   mode='train')

with open(pred_output, 'r') as fi:
    pred_dict = json.load(fi)
    pred_dict = get_latest_output(pred_dict)
assert len(examples) == len(pred_dict)

# for i in range(len(examples)):
#     print(i, len(examples[i].words), len(pred_dict[i]))
#     if len(examples[i].words) != len(pred_dict[i]):
#         token_check = [(len(tokenizer.tokenize(word)), word) for word in examples[i].words]
#         # print(token_check)
#         token_length = sum(l[0] for l in token_check)
#         if token_length > 510:
#             print('Warning', token_length)

all_tokens = [l for e in examples for l in e.labels]
counter = Counter(all_tokens)
print(counter)
#
# exit()

# for idx, lines in enumerate(zip(list_lines, list_lines_with_label)):
#     # if idx > 200:
#     #     break
#     print(idx)
#     export_path = os.path.join(output_path, 'sample_{:02d}.jpg'.format(idx))
#     plot_single(lines, export_path)

for idx, lines in enumerate(zip(examples, pred_dict)):
    # if idx > 50:
    #     break
    print("\n", idx)
    export_path = os.path.join(output_path, 'sample_{:02d}.jpg'.format(idx))
    plot_single_example(lines, export_path)
