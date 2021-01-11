import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from imutils import paths
from pprint import pprint
from transformers import AutoTokenizer
from matplotlib import pyplot as plt

from utils import bbox_string, actual_bbox_string, sort_funsd_reading_order

def preprocess_label_datapile(label):
    key_type = label['key_type']
    label = label['formal_key'].strip()
    
    if len(label) != 0 and key_type in ['key', 'value']:
        return key_type + '_' + label
    
    return 'other'


def process_label_invoice_categorized(label):
    key_type = label['key_type']
    label = label['formal_key'].strip()
    
    if key_type == 'key' or len(label) == 0:
        return 'other'
    if label in ['account_name', 'account_type', 'bank_name', 'branch_name', 
                 'company_department_name', 'document_number', 'invoice_number',
                 'item_unit']:
        return 'unkown'
    if label in ['account_number', 'amount_excluding_tax', 'amount_including_tax',
                 'item_line_number', 'item_quantity', 'item_quatity', 
                 'item_quantity_item_unit', 'item_total_amount', 'item_unit_amount', 
                 'item_total_excluding_tax', 'item_total_including_tax', 'tax']:
        return 'number'
    if label in ['company_address']:
        return 'address'
    if label in ['company_fax', 'company_tel']:
        return 'tel_fax'
    if label in ['company_zipcode']:
        return 'zipcode'
    if label in ['delivery_date', 'issued_date', 'payment_date']:
        return 'date'
    if label in ['item_name']:
        return 'description'
    return label


def process_label_invoice_full_class(label):
    key_type = label['key_type']
    label = label['formal_key'].strip()
    
    if key_type == 'key' or len(label) == 0:
        return 'other'
    if label in ['item_quantity', 'item_quatity']:
        return 'item_quantity'
    
    return label


def convert_one_datapile_to_funsd(data, image, tokenizer):
    width, height = image.size
    
    lines = []
    for line in data['attributes']['_via_img_metadata']['regions']:
        current_line = {}
        text = line['region_attributes']['label'].strip()
        if not text:
            continue
        
        if line['shape_attributes']['name'] == 'rect':
            x1 = line['shape_attributes']['x']
            y1 = line['shape_attributes']['y']
            line_width = line['shape_attributes']['width']
            line_height = line['shape_attributes']['height']
        elif line['shape_attributes']['name'] == 'polygon':
            x1 = min(line['shape_attributes']['all_points_x'])
            y1 = min(line['shape_attributes']['all_points_y'])
            line_width = max(line['shape_attributes']['all_points_x']) - x1
            line_height = max(line['shape_attributes']['all_points_y']) - y1
        
        if line_width < len(text):
            continue
            
        
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x1 + line_width, width)
        y2 = min(y1 + line_height, height)
        
        # if x2 < x1 or y2 < y1:
        #     from IPython import embed
        #     embed()
        
        current_line['text'] = text.replace('¥', '円')
        current_line['box'] = [x1, y1, x2, y2]
        current_line['label'] = preprocess_label_datapile(line['region_attributes'])
        
        words = []
        start_x = x1
        tokens = [tokenizer.unk_token] if text == 'NotValid' else tokenizer.tokenize(text)
        token_width = int(line_width / len(tokens))
        for w in tokens:
            current_word = {}
            
            # if len(text) == 1:
            #     token_width = round(1 / len(tokens) * line_width)
            # elif w == tokenizer.unk_token:
            #     token_width = round(1 / len(text) * line_width)
            # else:
            #     token_width = round(len(w.replace('##', '', 1)) / len(text) * line_width)
            
            start_x = min(start_x, x2)
            end_x = min(x2, start_x + token_width - 1)
            if start_x > end_x or y1 > y2:
                print(line_width, tokenizer.tokenize(text))
                print(token_width)
                print('#' * 60, 'ERROR')
                print(text, w, sep='|')
                import IPython
                IPython.embed()
            current_word['text'] = w
            current_word['box'] = [start_x, y1, end_x, y2]
            # print([start_x, y1, end_x, y2])
            start_x = end_x + 1
            words.append(current_word)
        # print('------')
        
        current_line['words'] = words
        
        lines.append(current_line)
    
    # lines = sorted(lines, key=lambda x : (x['box'][1], x['box'][0]))
    lines = sort_funsd_reading_order(lines)
    
    return lines


def convert_datapile_to_funsd(args):
    data_map = {}
    for p in paths.list_images(args.data_dir):
        k = os.path.splitext(os.path.basename(p))[0]
        data_map[k] = {'image': p}
    
    for p in paths.list_files(args.data_dir, validExts=('.json')):
        k = os.path.splitext(os.path.basename(p))[0]
        if k in data_map:
            data_map[k]['label'] = p
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              do_lower_case=True)
    
    annotations = {}
    if args.visualize:
        visualization_dir = Path(args.output_dir) / 'visualization'
        visualization_dir.mkdir(parents=True, exist_ok=True)
    # for sample in list(data_map.values())[:3]:
    for sample in data_map.values():
        if 'label' not in sample or 'image' not in sample:
            pprint(sample)
            continue
        
        label_path = sample['label']
        image_path = sample['image']
        
        file_name = os.path.basename(image_path)
        # if '0785_070_16' not in file_name:
        #     continue
        print()
        print('-' * 100)
        print(file_name)
        
        current_item = {}
        current_item['image_path'] = image_path
        
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        print(label_path)
        with open(label_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        
        current_item['form'] = convert_one_datapile_to_funsd(data, image, tokenizer)
        annotations[file_name] = current_item
        
        # from pprint import pprint
        # pprint(kv_label)
        if args.visualize:
            image = np.array(image, dtype=np.float32)
            if np.max(image) <= 1.0:
                image = image * 255
            for line in current_item['form']:
                if line['label'] != '':
                    x1, y1, x2, y2 = line['box']
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255 , 0), thickness=7)
                    cv2.putText(image, line['label'], (x1, y1), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
            # cv2.imwrite(str(visualization_dir / file_name), image)
            image = Image.fromarray(image.astype(np.uint8))
            image.save(str(visualization_dir / file_name))
            # plt.imshow(image)
            # plt.show()
        # from IPython import embed
        # embed()
    
    return annotations


def convert(annotations, agrs):
    if args.so_only:
        b_tag = 'S-'
        i_tag = 'S-'
        e_tag = 'S-'
    else:
        b_tag = 'B-'
        i_tag = 'I-'
        e_tag = 'E-'
    
    fw, fbw, fiw = '', '', ''
    err = {}
    for file_name, data in annotations.items():
        token_cnt = 0
        image_path = data['image_path']
        image = Image.open(image_path)
        width, length = image.size
        for item in data["form"]:
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
    
            if len(words) == 0:
                continue
    
            current_len = len(words)
    
            if token_cnt + current_len > args.max_len:
                fw += '\n'
                fbw += '\n'
                fiw += '\n'
                token_cnt = 0
    
            if label == "other":
                for w in words:
                    if int(w['box'][0] / width * 1000) == int(w['box'][2] / width * 1000):
                        if file_name in err:
                            err[file_name].append({
                                'line': item,
                                'w': width,
                                'h': length
                            })
                        else:
                            err[file_name] = [{
                                'line': item,
                                'w': width,
                                'h': length
                            }]
                    fw += (w["text"] + "\tO\n")
                    fbw += (
                        w["text"]
                        + "\t"
                        + bbox_string(w["box"], width, length)
                        + "\n"
                    )
                    fiw += (
                        w["text"]
                        + "\t"
                        + actual_bbox_string(w["box"], width, length)
                        + "\t"
                        + file_name
                        + "\n"
                    )
            else:
                if len(words) == 1:
                    fw += (words[0]["text"] + "\tS-" + label.upper() + "\n")
                    fbw += (
                        words[0]["text"]
                        + "\t"
                        + bbox_string(words[0]["box"], width, length)
                        + "\n"
                    )
                    fiw += (
                        words[0]["text"]
                        + "\t"
                        + actual_bbox_string(words[0]["box"], width, length)
                        + "\t"
                        + file_name
                        + "\n"
                    )
                else:
                    fw += (words[0]["text"] + "\t" + b_tag + label.upper() + "\n")
                    fbw += (
                        words[0]["text"]
                        + "\t"
                        + bbox_string(words[0]["box"], width, length)
                        + "\n"
                    )
                    fiw += (
                        words[0]["text"]
                        + "\t"
                        + actual_bbox_string(words[0]["box"], width, length)
                        + "\t"
                        + file_name
                        + "\n"
                    )
                    for w in words[1:-1]:
                        fw += (w["text"] + "\t" + i_tag + label.upper() + "\n")
                        fbw += (
                            w["text"]
                            + "\t"
                            + bbox_string(w["box"], width, length)
                            + "\n"
                        )
                        fiw += (
                            w["text"]
                            + "\t"
                            + actual_bbox_string(w["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    fw += (words[-1]["text"] + "\t" + e_tag + label.upper() + "\n")
                    fbw += (
                        words[-1]["text"]
                        + "\t"
                        + bbox_string(words[-1]["box"], width, length)
                        + "\n"
                    )
                    fiw += (
                        words[-1]["text"]
                        + "\t"
                        + actual_bbox_string(words[-1]["box"], width, length)
                        + "\t"
                        + file_name
                        + "\n"
                    )
    
            token_cnt += current_len
    
        if token_cnt != 0:
            fw += ("\n")
            fbw += ("\n")
            fiw += ("\n")
    
    
    with open(os.path.join(args.output_dir, args.data_split + ".txt"), "w",
              encoding="utf8") as f:
        f.write(fw)
    
    with open(os.path.join(args.output_dir, args.data_split + "_box.txt"), "w",
              encoding="utf8") as f:
        f.write(fbw)
    
    with open(os.path.join(args.output_dir, args.data_split + "_image.txt"), "w",
              encoding="utf8") as f:
        f.write(fiw)
    
    if args.data_split == 'train':
        label_list = set()
        for line in open(os.path.join(args.output_dir, 'train.txt'), 
                         'r', encoding='utf8').read().split('\n'):
            if not line:
                continue
            label_list.add(line.split('\t')[-1])
        
        label_list = sorted(list(label_list))
        # from IPython import embed
        # embed()
        with open(os.path.join(args.output_dir, 'labels.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(label_list))
        
    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=r"data_raw\tmp"
    )
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default=r"data_raw\tmp")
    parser.add_argument("--model_name_or_path", type=str, default='cl-tohoku/bert-base-japanese')
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--so_only", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.data_dir + '_layoutlm'
    
    from pathlib import Path
    p = Path(args.output_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    
    annotations = convert_datapile_to_funsd(args)
    
    err = convert(annotations, args)