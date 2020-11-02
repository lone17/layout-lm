import os
import json
import math
import random
import argparse
from queue import PriorityQueue

import cv2
import editdistance
import numpy as np
from PIL import Image
from imutils import paths
from transformers import AutoTokenizer
from matplotlib import pyplot as plt

def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def find_total(lines, total_value):
    def on_left_or_top(this, that):
        if this == that:
            return True
        
        if min(this['box'][3], that['box'][3]) > max(this['box'][1], that['box'][1]):
            return this['box'][2] < that['box'][0]
        
        if min(this['box'][2], that['box'][2]) > max(this['box'][0], that['box'][0]):
            return 0 < that['box'][1] - this['box'][3] < that['box'][3] - that['box'][1]
        
        return False
    
    def kv_distance(key, value):
        if not on_left_or_top(key, value):
            return 10e6
        
        key_center = (key['box'][0] + key['box'][2]) / 2, (key['box'][1] + key['box'][3]) / 2
        value_center = (value['box'][0] + value['box'][2]) / 2, (value['box'][1] + value['box'][3]) / 2
        
        return min(abs(key_center[1] - value_center[1]),
                   abs(key_center[0] - value_center[0]),
                   abs(key['box'][2] - value['box'][2]),
                   abs(key['box'][0] - value['box'][0]))
    
    def distance(this, that):
        this_center = (this['box'][0] + this['box'][2]) / 2, (this['box'][1] + this['box'][3]) / 2
        that_center = (that['box'][0] + that['box'][2]) / 2, (that['box'][1] + that['box'][3]) / 2
        
        return math.sqrt((this_center[0] - that_center[0])**2 
                         + (this_center[1] - that_center[1])**2)
    
    value_candidates = []
    key_candidates = []
    for line in lines:
        if total_value.replace(' ', '').replace('.', ',') in line['text'].replace(' ', '').replace('.', ','):
            value_candidates.append(line)
        if 'total' in line['text'].lower():
            key_candidates.append(line)
    
    value_candidates_w_key = []
    value_candidates_wo_key = []
    for i, value_line in enumerate(value_candidates):
        min_distance = 10e6
        for key_line in key_candidates:
            d = kv_distance(key_line, value_line)
            if d < min_distance:
                min_distance = d
        if min_distance < 10e6:
            value_candidates_w_key.append((value_line, min_distance))
        else:
            value_candidates_wo_key.append(value_line)
    
    value_candidates_w_key = [x for x in value_candidates_w_key if x[1] < 10e6]
    
    value_candidates_w_key = sorted(value_candidates_w_key, 
                                    key=lambda x : x[1])
    
    if len(value_candidates_w_key) > 0:
        # for line, _ in value_candidates_w_key[:3]:
        #     line['label'] = 'w_key'
        # from pprint import pprint
        # print('w key')
        # pprint(value_candidates_w_key)
            
        if len(value_candidates_w_key) == 1:
            return value_candidates_w_key[0][0]
        
        lowest_line = sorted(value_candidates_w_key[:3], 
                             key=lambda x : x[0]['box'][1])[-1]
        
        
        return lowest_line[0]
    elif len(value_candidates_wo_key) > 0:
        # for line in value_candidates_wo_key:
        #     line['label'] = 'wo_key'
        # from pprint import pprint
        # print('wo key')
        # pprint(value_candidates_wo_key)
            
        if len(value_candidates_wo_key) == 1:
            return value_candidates_wo_key[0]
        
        value_candidates_wo_key = [(line, line['box'][3] - line['box'][1]) 
                                   for line in value_candidates_wo_key]
        value_candidates_wo_key = sorted(value_candidates_wo_key, 
                                         key=lambda x : x[1],
                                         reverse=True)
        
        if value_candidates_wo_key[0][1] > 1.1 * value_candidates_wo_key[1][1]:
            return value_candidates_wo_key[0][0]
        
        lowest_line = sorted(value_candidates_wo_key, 
                             key=lambda x : x[0]['box'][1])[-1]
        
        return lowest_line[0]

def add_kv_label(lines, kv_label):
    # sort lines by reading order
    lines = sorted(lines, key=lambda x: (x['box'][1], x['box'][0]))
    
    pos_list = []
    whole_text = ''
    for line in lines:
        text = line['text'].replace(' ', '').replace(',', '.')
        pos_list.append(len(whole_text))
        whole_text += text
    pos_list.append(len(whole_text))
    
    remaining_kv = {}
    for k, v in kv_label.items():
        if k.lower() == 'total':
            total_line = find_total(lines, v)
            if total_line is not None:
                total_line['label'] = k
                continue
            
        text = v.replace(' ', '').replace(',', '.')
        start = whole_text.find(text)
        if start >= 0:
            for i, pos in enumerate(pos_list[:-1]):
                if start <= pos < start + len(text):
                    lines[i]['label'] = k
                elif pos < start < pos_list[i+1]:
                    lines[i]['label'] = k
                elif pos >= start + len(text):
                    break
        else:
            remaining_kv[k] = v
    
    # print(whole_text)
    print('remaining', remaining_kv)
    # from IPython import embed
    # embed()
    
    still_remaining_kv = {}
    for k, v in remaining_kv.items():
        text = tuple(v.replace(',', '.').split())
        found_something = False
        for line in lines:
            tmp_text = text
            cnt = 0
            num_words = len(line['words'])
            for w in line['text'].replace(',', '.').split():
                if w in tmp_text:
                    idx = tmp_text.index(w)
                    cnt += 1
                    tmp_text = tmp_text[idx + 1:]
            if cnt / num_words >= 2/3 and (num_words - cnt) <= 3:
                found_something = True
                line['label'] = k
                
        if not found_something:
            still_remaining_kv[k] = v
    
    # embed()
    print('still remaining', still_remaining_kv)
    # naive way: not work with multi-line fields
    for k, v in still_remaining_kv.items():
        best_match = None
        best_distance = 1e6
        for line in lines:
            distance = editdistance.eval(v, line['text'])
            if distance < best_distance:
                best_match = line
                best_distance = distance
        best_match['label'] = k
        if best_distance != 0:
            print(k, v, best_match['text'], sep=' | ')
    # embed()
    
    return lines


def merge_line_by_key(lines):
    def merge_box(box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        
        return [x1, y1, x2, y2]
    
    new_lines = []
    
    fields = {}
    
    for line in lines:
        label = line['label']
        if label == 'other':
            new_lines.append(line)
        else:
            if label not in fields:
                fields[label] = [line]
            else:
                fields[label].append(line)
    
    for k, v in fields.items():
        current_field = v[0]
        for l in v[1:]:
            current_field['text'] += ' ' + l['text']
            current_field['words'] += l['words']
            current_field['box'] = merge_box(current_field['box'], l['box'])
        current_field['words'] = sorted(current_field['words'], 
                                        key=lambda x: (x['box'][1], x['box'][0]))
        new_lines.append(current_field)
    
    # re-sort lines by reading order
    new_lines = sorted(new_lines, key=lambda x: (x['box'][1], x['box'][0]))
    
    return new_lines


def convert_sroie_to_funsd(data_dir, output_dir, visualize):
    annotations = {}
    from pathlib import Path
    visualization_dir = Path(output_dir) / 'visualization'
    visualization_dir.mkdir(parents=True, exist_ok=True)
    for image_path in paths.list_images(os.path.join(data_dir, '0325updated.task2train(626p)')):
        if ').' in os.path.basename(image_path):
            continue
        # if 'X51006414485' not in os.path.basename(image_path):
        #     continue
        file_name = os.path.basename(image_path)
        print()
        print('-' * 100)
        print(file_name)
        kv_label_path = image_path.replace(os.path.splitext(image_path)[-1], '.txt')
        text_label_path = kv_label_path.replace('task2', 'task1')
        
        current_item = {}
        current_item['image_path'] = image_path
        
        image = Image.open(image_path)
        width, height = image.size
        
        lines = []
        for line in open(text_label_path, 'r', encoding='utf8').read().split('\n'):
            current_line = {}
            line = line.strip()
            if not line:
                continue
            *pos, text = line.split(',', 8)
            pos = [int(i) for i in pos]
            
            x1, y1 = pos[0], pos[1]
            x2, y2 = pos[4], pos[5]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            line_width = x2 - x1
            
            current_line['text'] = text
            current_line['box'] = [x1, y1, x2, y2]
            current_line['label'] = 'other'
            
            words = []
            start_x = x1
            for w in text.split():
                current_word = {}
                word_width = int((len(w) + 1) / len(text) * line_width)
                end_x = min(x2, start_x + word_width)
                current_word['text'] = w
                current_word['box'] = [start_x, y1, end_x, y2]
                start_x = start_x + 1
                words.append(current_word)
            
            current_line['words'] = words
            
            lines.append(current_line)
        
        kv_label = json.load(open(kv_label_path, 'r', encoding='utf8'))
        
        lines = add_kv_label(lines, kv_label)
        lines = merge_line_by_key(lines)
        current_item['form'] = lines
        
        annotations[file_name] = current_item
        
        # from pprint import pprint
        # pprint(kv_label)
        if visualize:
            image = np.array(image)
            for line in current_item['form']:
                if line['label'] != 'other':
                    x1, y1, x2, y2 = line['box']
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0 , 0))
                    cv2.putText(image, line['label'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            cv2.imwrite(str(visualization_dir / file_name), image)
            # plt.imshow(image)
            # plt.show()
    
    return annotations


def convert(annotations, output_dir, data_split):
    with open(
        os.path.join(output_dir, data_split + ".txt.tmp"),
        "w",
        encoding="utf8",
    ) as fw, open(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file_name, data in annotations.items():
            image_path = data['image_path']
            image = Image.open(image_path)
            width, length = image.size
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        fw.write(w["text"] + "\tO\n")
                        fbw.write(
                            w["text"]
                            + "\t"
                            + bbox_string(w["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            w["text"]
                            + "\t"
                            + actual_bbox_string(w["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                else:
                    if len(words) == 1:
                        fw.write(words[0]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    else:
                        fw.write(words[0]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                        for w in words[1:-1]:
                            fw.write(w["text"] + "\tS-" + label.upper() + "\n")
                            fbw.write(
                                w["text"]
                                + "\t"
                                + bbox_string(w["box"], width, length)
                                + "\n"
                            )
                            fiw.write(
                                w["text"]
                                + "\t"
                                + actual_bbox_string(w["box"], width, length)
                                + "\t"
                                + file_name
                                + "\n"
                            )
                        fw.write(words[-1]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[-1]["text"]
                            + "\t"
                            + bbox_string(words[-1]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[-1]["text"]
                            + "\t"
                            + actual_bbox_string(words[-1]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
            fw.write("\n")
            fbw.write("\n")
            fiw.write("\n")


def seg_file(file_path, tokenizer, max_len):
    subword_len_counter = 0
    output_path = file_path[:-4]
    with open(file_path, "r", encoding="utf8") as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        for line in f_p:
            line = line.rstrip()

            if not line:
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue
            token = line.split("\t")[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n" + line + "\n")
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len

            fw_p.write(line + "\n")


def seg(model_name_or_path, output_dir, data_split, max_len):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, do_lower_case=True
    )
    seg_file(
        os.path.join(output_dir, data_split + ".txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_box.txt.tmp"),
        tokenizer,
        max_len,
    )
    seg_file(
        os.path.join(output_dir, data_split + "_image.txt.tmp"),
        tokenizer,
        max_len,
    )
    
    if data_split == 'train':
        label_list = set()
        for line in open(os.path.join(output_dir, 'train.txt'), 
                         'r', encoding='utf8').read().split('\n'):
            if not line:
                continue
            label_list.add(line.split('\t')[-1])
        
        label_list = sorted(list(label_list))
        # from IPython import embed
        # embed()
        with open(os.path.join(output_dir, 'labels.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(label_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=r"D:\Experiments\layout-lm\SROIE2019"
    )
    parser.add_argument("--output_dir", type=str, default="sroie_with_SO")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train_size", type=float, default=1.0)
    args = parser.parse_args()
    
    from pathlib import Path
    p = Path(args.output_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    annotations = convert_sroie_to_funsd(args.data_dir, args.output_dir,
                                         visualize=False)
    key_list = list(annotations.keys())
    random.shuffle(key_list)
    random.seed(args.seed)
    split_point = int(args.train_size * len(key_list))
    
    convert({
                k: v for k, v in annotations.items() 
                if k in key_list[:split_point]
            }, 
            args.output_dir, 'train')
    seg(args.model_name_or_path, args.output_dir, 'train', args.max_len)
    
    if args.train_size < 1:
        convert({
                    k: v for k, v in annotations.items() 
                    if k in key_list[split_point:]
                }, 
                args.output_dir, 'val')
        seg(args.model_name_or_path, args.output_dir, 'val', args.max_len)
