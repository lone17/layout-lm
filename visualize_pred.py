import os
import json

import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    WEIGHTS_NAME,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    BertModel,
    BertForTokenClassification,
    LayoutLMTokenizer, 
    LayoutLMForTokenClassification,
    LayoutLMModel,
    LayoutLMConfig,
    AdamW,
    get_linear_schedule_with_warmup
)

from datasets import InputExample, convert_examples_to_features
from preprocess_datapile import(process_label_invoice_full_class,
                                convert_one_datapile_to_funsd)
from utils import sort_funsd_reading_order

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i - 10, coordinates[0][1] - i - 10)
        rect_end = (coordinates[1][0] + i + 10, coordinates[1][1] + i + 10)
        draw.rectangle((rect_start, rect_end), outline=color)


def get_examples_from_one_sample(args, image, annotation, tokenizer):
    def normalize_box(box):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]
    
    funsd_annotation = convert_one_datapile_to_funsd(annotation, image, tokenizer)
    funsd_annotation = sort_funsd_reading_order(funsd_annotation)
    
    width, height = image.size
    
    examples = []
    token_cnt = 0
    
    tokens = []
    boxes = []
    actual_bboxes = []
    labels = []
    
    for item in funsd_annotation:
        words, label = item["words"], item["label"]
        words = [w for w in words if w["text"].strip() != ""]
        for w in words:
            
            if len(words) == 0:
                continue
            
            current_len = len(words)
            
            if token_cnt + current_len > args.max_seq_length - 2:
                examples.append(
                    InputExample(
                        guid="%s-%d".format('test', 1),
                        words=tokens,
                        labels=labels,
                        boxes=boxes,
                        actual_bboxes=actual_bboxes,
                        file_name='test',
                        page_size=[width, height],
                    )
                )
                
                token_cnt = 0
                
                tokens = []
                boxes = []
                actual_bboxes = []
                labels = []
            
            tokens.append(w['text'])
            labels.append("O")
            actual_bboxes.append(w['box'])
            boxes.append(normalize_box(w['box']))
            
            token_cnt += current_len
            
        if token_cnt > 0:
            examples.append(
                InputExample(
                    guid="%s-%d".format('test', 1),
                    words=tokens,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name='test',
                    page_size=[width, height],
                )
            )
    
    
    features = convert_examples_to_features(
        examples,
        None,
        args.max_seq_length,
        tokenizer,
        args.is_tokenized,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        pad_token_label_id=args.pad_token_label_id,
    )
    
    return features


def visualize_label(args, image, annotation, fields):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", 15)
    
    regions = []
    for line in annotation['attributes']['_via_img_metadata']['regions']:
        if line['shape_attributes']['name'] == 'rect':
            x1 = line['shape_attributes']['x']
            y1 = line['shape_attributes']['y']
            x2 = x1 + line['shape_attributes']['width']
            y2 = y1 + line['shape_attributes']['height']
        elif line['shape_attributes']['name'] == 'polygon':
            x1 = min(line['shape_attributes']['all_points_x'])
            y1 = min(line['shape_attributes']['all_points_y'])
            x2 = max(line['shape_attributes']['all_points_x']) - x1
            y2 = max(line['shape_attributes']['all_points_y']) - y1
        
        label = process_label_invoice_full_class(line['region_attributes'])
        
        regions.append({'box': [x1, y1, x2, y2], 'label': label})
    
    regions = sort_funsd_reading_order(regions)
    
    prev_pos = None
    for r in regions:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        if label != 'other':
            draw.text((x1, y1), label + ' ({:d})'.format(fields.index(label.upper())), 
                      fill='brown', font=font, thickness=1)
        # draw_rectangle(draw, ((x1, y1), (x2, y2)), 'black', width=6)
        draw.line((x1, y1) + (x2, y1), fill='brown', width=3)
        if prev_pos is not None:
            draw.line(prev_pos + (x1, y1), fill='brown', width=1)
        prev_pos = (x2, y1)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255 , 0), thickness=3)
        # if label != 'other':
        #     cv2.putText(image, label, (x1, y1), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
    
    return image


def visualize_prediction(args, image, predictions, boxes, fields):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", 15)
    
    for i, f in enumerate(fields):
        print(i, '-', f)
    
    prev_pos = None
    prev_tag = ''
    current_field = None
    for p, [x1, y1, x2, y2] in zip(predictions, boxes):
        if p.startswith('B-'):
            color = 'green'
            current_field = p.split('-')[-1]
        elif p.startswith('I-'):
            color = 'blue'
            current_field = p.split('-')[-1]
        elif p.startswith('E-'):
            color = 'red'
            current_field = None
        elif p.startswith('S-'):
            color = 'yellow'
            current_field = None
        elif p == 'O':
            current_field = None
            color = 'gray'
        else:
            raise Error('Unknown class ' + p)
        
        if p.startswith('B-') or (p != 'O' and p.split('-')[-1] != current_field):
            draw.text((x1, y2 + 3), p[:2] + str(fields.index(p[2:])), fill=color, 
                      font=font, thickness=1)
        if prev_pos is not None and prev_tag == p and p != 'O':
            draw.line(prev_pos + (x1, y2), fill=color, width=3)
        draw.line((x1, y2, x2, y2), fill=color, width=3)
        # draw_rectangle(draw, ((x1, y1), (x2, y2)), color, width=6)
        prev_pos = (x2, y2)
        prev_tag = p
    
    return image


def predict_one_sample(args, image, annotation, labels, model, tokenizer):
    label_map = {i: label for i, label in enumerate(labels)}
    
    features = get_examples_from_one_sample(args, image, annotation, tokenizer)
    
    predictions = []
    boxes = []
    model.eval()
    for f in features:
        with torch.no_grad():
            if args.bert_model is not None:
                inputs = {
                    "input_ids": torch.tensor([f.input_ids], dtype=torch.long).to(args.device),
                    "attention_mask": torch.tensor([f.attention_mask], dtype=torch.long).to(args.device),
                    "token_type_ids": torch.tensor([f.segment_ids], dtype=torch.long).to(args.device),
                }
            else:
                inputs = {
                    "input_ids": torch.tensor([f.input_ids], dtype=torch.long).to(args.device),
                    "attention_mask": torch.tensor([f.attention_mask], dtype=torch.long).to(args.device),
                    "token_type_ids": torch.tensor([f.segment_ids], dtype=torch.long).to(args.device),
                    "bbox": torch.tensor([f.boxes], dtype=torch.long).to(args.device)
                }
            
            outputs = model(**inputs)
            
            logits = outputs.logits
            pred = logits.detach().cpu().numpy()
            pred = np.argmax(pred, axis=2)[0]
            
            assert len(pred) == len(f.label_ids)
            assert len(pred) == len(f.actual_bboxes)
            
            for i, label_id in enumerate(f.label_ids):
                if label_id != args.pad_token_label_id:
                    predictions.append(label_map[pred[i]])
                    boxes.append(f.actual_bboxes[i])
    
    return predictions, boxes


def process_one_sample(args, image_path, label_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    with open(label_path, 'r', encoding='utf8') as f:
        annotation = json.load(f)
    
    labels = get_labels(os.path.join(args.data_dir, 'labels.txt'))
    fields = sorted(list(set([p.split('-')[-1] for p in labels])))
    
    if args.bert_model is None:
        tokenizer = LayoutLMTokenizer.from_pretrained(args.layoutlm_model,
                                                      do_lower_case=True)
        model = LayoutLMForTokenClassification.from_pretrained(args.layoutlm_model,
                                                               return_dict=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        model = BertForTokenClassification.from_pretrained(args.bert_model,
                                                           return_dict=True)
    
    model.to(args.device)
    
    predictions, boxes = predict_one_sample(args, image, annotation, labels, 
                                            model, tokenizer)
    
    image = visualize_label(args, image, annotation, fields)
    image = visualize_prediction(args, image, predictions, boxes, fields)
    
    return image


args = dict(
    data_dir='data_processed/sroie_multiline_SO_with_val',
    max_seq_length=512,
    # layoutlm_model='D:\Experiments\layout-lm\ep-191-val_loss-0.64-val_f1-0.81-train_loss-0.00-train_f1-1.00',
    bert_model=None,
    model_type='layoutlm',
    train_batch_size=2,
    eval_batch_size=16,
    device='cuda',
    so_only=True,
    is_tokenized=False,
    bert_only=True,
    retrain_word_embedder=False,
    retrain_layout_embedder=False,
    pad_token_label_id=nn.CrossEntropyLoss().ignore_index
)

# For invoice
args.update(dict(
    bert_model='bert_only\ep-70-val_loss-0.61-val_f1-0.76-train_loss-0.00-train_f1-1.00',
    data_dir='data_processed/invoice3_read_order_full_class',
    is_tokenized=True,
))

class Args:
    def __init__(self, args):
        self.__dict__ = args

args = Args(args)


image_path = r'data_raw\invoice3\test\test_338_files\images\10_0.png'
label_path = r'data_raw\invoice3\test\test_338_files\labels\10_0.json'

image = process_one_sample(args, image_path, label_path)
image.show()
