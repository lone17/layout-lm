import os
import json
from pathlib import Path

import cv2
import torch
import numpy as np
from torch import nn
from imutils import paths
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
from seqeval.scheme import Entities, IOBES, IOB2, IOE2

from datasets import InputExample, convert_examples_to_features
from preprocess_datapile import (preprocess_label_datapile,
                                 process_label_invoice_full_class,
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


def get_examples_from_one_sample(args, image, annotation, tokenizer, datapile_format=True):
    def normalize_box(box):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    funsd_annotation = convert_one_datapile_to_funsd(annotation, image, tokenizer, datapile_format=datapile_format)

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

        for w in words:
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


def visualize_label(image, annotation, fields):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Dengb.ttf", 15)

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

        label = preprocess_label_datapile(line['region_attributes'])

        regions.append({'box': [x1, y1, x2, y2], 'label': label})

    regions = sort_funsd_reading_order(regions)

    prev_pos = None
    for r in regions:
        x1, y1, x2, y2 = r['box']
        label = r['label']
        if label.upper() not in fields:
            label = 'other'
        if label != 'other':
            draw.text((x1, y1), label + ' ({:d})'.format(fields.index(label.upper())),
                      fill='brown', font=font, thickness=1)
            # draw_rectangle(draw, ((x1, y1), (x2, y2)), 'black', width=6)
            draw.line((x1, y1) + (x2, y1), fill='brown', width=3)
        # if prev_pos is not None:
        #     draw.line(prev_pos + (x1, y1), fill='brown', width=1)
        prev_pos = (x2, y1)
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255 , 0), thickness=3)
        # if label != 'other':
        #     cv2.putText(image, label, (x1, y1),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)

    return image


def write_output_kv(output_path, filename, entities):
    report_path = os.path.join(output_path, filename)
    if not os.path.isdir(report_path):
        os.mkdir(report_path)
    report_dict = []
    for entity in entities:
        x1, y1, x2, y2 = entity['box']
        split_index = entity['tag'].find('_')
        key_type, formal_key = entity['tag'][:split_index].lower(), entity['tag'][split_index + 1:].lower()
        if key_type != 'value':
            continue
        text = entity['text'].replace(' ', '').replace('#', '').replace('[UNK]', '').upper()
        print(formal_key, text)
        if text is None:
            continue
        report_dict.append({
            'location': [[x1, y1], [x1, y2], [x2, y2], [x2, y1]],
            'text': text,
            'key_type': 'value',
            'type': formal_key,
            'formal_key': formal_key,
            'confidence': 0.5,
            'bbox': entity['box'],
            'box': entity['box']
        })

    with open(os.path.join(report_path, 'kv.json'), 'w') as fo:
        json.dump(regions, fo, indent=4, ensure_ascii=False)


def visualize_prediction(image, predictions, entities, boxes, fields):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Dengb.ttf", 15)

    prev_pos = None
    prev_tag = ''
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
            color = 'magenta'
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
        if p != 'O':
            draw.line((x1, y2, x2, y2), fill=color, width=3)
        prev_pos = (x2, y2)
        prev_tag = p

    for entity in entities:
        list_x, list_y = [], []
        for box in boxes[entity['start']: entity['end']]:
            x1, y1, x2, y2 = box
            list_x.extend((x1, x2))
            list_y.extend((y1, y2))
        min_x, min_y, max_x, max_y = min(list_x), min(list_y), max(list_x), max(list_y)
        draw_rectangle(draw, ((min_x + 8, min_y + 8), (max_x - 8, max_y - 8)), color='blue', width=4)
        draw.text((min_x, max_y + 13), '{}'.format(entity['text']), fill='blue',
                  font=font, thickness=2)

    return image


def hide_eng_tag(predictions):
    predictions = [tag[:2] + tag[2:].split('_')[0] for tag in predictions]
    new_predictions = ['I' + tag[1:] if tag[:1] in ('E', 'S') else tag for tag in predictions]
    return new_predictions


def hide_begin_tag(predictions):
    predictions = [tag[:2] + tag[2:].split('_')[0] for tag in predictions]
    new_predictions = ['I' + tag[1:] if tag[:1] in ('B', 'S') else tag for tag in predictions]
    return new_predictions


def predict_one_sample(args, filename, image, annotation, labels, model, tokenizer, datapile_format=False):
    label_map = {i: label for i, label in enumerate(labels)}

    features = get_examples_from_one_sample(args, image, annotation, tokenizer, datapile_format=datapile_format)

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

            inputs_all["input_ids"].extend(f.input_ids)
            inputs_all["bboxes"].extend(f.boxes)

            outputs = model(**inputs)
            
            logits = outputs.logits
            pred = logits.detach().cpu().numpy()
            pred = np.argmax(pred, axis=2)[0]
            
            assert len(pred) == len(f.label_ids)
            assert len(pred) == len(f.actual_bboxes)
            
            try:
                for i, label_id in enumerate(f.label_ids):
                    if label_id >= 0:
                        predictions.append(label_map[pred[i]])
                        boxes.append(f.actual_bboxes[i])
                        input_ids.append((f.input_ids[i]))
            except:
                from IPython import embed
                embed()

    schemes = [(IOBES, None), (IOB2, hide_eng_tag), (IOE2, hide_begin_tag), ]
    entities = []

    inputs = input_ids
    predictions_origin = predictions.copy()
    boxes_origin = boxes.copy()

    for scheme, preprocess_function in schemes:
        if preprocess_function is not None:
            tmp_predictions = preprocess_function(predictions)
        else:
            tmp_predictions = predictions

        mask = [False] * len(predictions)
        sequences = Entities([tmp_predictions], scheme=scheme)

        for entity in sequences.entities[0]:
            entity_text = tokenizer.decode(inputs[entity.start: entity.end])
            for _id in range(entity.start, entity.end):
                mask[_id] = True

            list_x, list_y = [], []
            for box in boxes[entity.start: entity.end]:
                x1, y1, x2, y2 = box
                list_x.extend((x1, x2))
                list_y.extend((y1, y2))

            if scheme == IOB2:
                tag = predictions[entity.start][2:]
            elif scheme == IOE2:
                tag = predictions[entity.end-1][2:]
            else:
                tag = entity.tag

            min_x, min_y, max_x, max_y = min(list_x), min(list_y), max(list_x), max(list_y)
            entity_box = [min_x + 8, min_y + 8, max_x - 8, max_y - 8]

            if entity.end - entity.start > 1 or scheme == IOBES:
                # print(entity_text, tag)
                entities.append({
                    'text': entity_text,
                    'start': entity.start,
                    'end': entity.end,
                    'tag': tag,
                    'box': entity_box
                })

        new_predictions = []
        new_boxes = []
        new_inputs = []
        for _id, status in enumerate(mask):
            if not status:
                new_predictions.append(predictions[_id])
                new_boxes.append(boxes[_id])
                new_inputs.append(inputs[_id])
            else:
                new_predictions.append('O')
                new_boxes.append([0, 0, 0, 0])
                new_inputs.append('')
        predictions = new_predictions
        boxes = new_boxes
        inputs = inputs

    write_output_kv(args.kv_output_dir, filename, entities)
    # print(len(predictions_origin), len(boxes_origin))

    return entities, predictions_origin, boxes_origin


def process_one_sample(args, filename, image_path, label_path, pred_path, model, tokenizer, labels):
    image = Image.open(image_path)
    image = image.convert('RGB')

    with open(label_path, 'r', encoding='utf8') as f:
        annotation = json.load(f)

    with open(pred_path, 'r', encoding='utf8') as f:
        pred = json.load(f)

    fields = sorted(list(set([p.split('-')[-1] for p in labels])))

    entities, predictions, boxes = predict_one_sample(args, filename, image, pred, labels,
                                                      model, tokenizer, datapile_format=False)

    image = visualize_label(image, annotation, fields)
    image = visualize_prediction(image, predictions, entities, boxes, fields)

    return image, predictions


def process(args):
    data_map = {}
    for p in paths.list_images(args.raw_data_dir):
        k = os.path.splitext(os.path.basename(p))[0]
        data_map[k] = {'image': p}

    for p in paths.list_files(args.raw_data_dir, validExts=('.json')):
        k = os.path.splitext(os.path.basename(p))[0]
        if k in data_map:
            data_map[k]['label'] = p

    for p in paths.list_files(args.pred_data_dir, validExts=('.json')):
        if os.path.basename(p) != 'ocr_output.json':
            continue
        k = os.path.basename(os.path.dirname(p))
        if k in data_map:
            data_map[k]['pred'] = p

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    model = LayoutLMForTokenClassification.from_pretrained(args.layoutlm_model,
                                                           return_dict=True)

    model.to(args.device)

    labels = get_labels(os.path.join(args.processed_data_dir, 'labels.txt'))

    pred_dict = {}
    for k, v in data_map.items():
        if 'image' not in v or 'label' not in v or 'pred' not in v:
            # print('error', v)
            continue

        print(k)

        out_image, preds = process_one_sample(args, k, v['image'], v['label'], v['pred'], model, tokenizer, labels)

        out_image.save(os.path.join(args.visualization_dir, k + '.png'))

        pred_dict[k] = preds

    with open(os.path.join(args.visualization_dir, 'preds.json'), 'w',
              encoding='utf-8') as f:
        json.dump(pred_dict, f, indent=4, ensure_ascii=False)


args = dict(
    visualization_dir=None,
    max_seq_length=512,
    model_type='layoutlm',
    train_batch_size=2,
    eval_batch_size=16,
    device='cuda',
    so_only=True,
    bert_only=True,
    retrain_word_embedder=False,
    retrain_layout_embedder=False,
    pad_token_label_id=nn.CrossEntropyLoss().ignore_index
)

# For invoice
args.update(dict(
    layoutlm_model='output_sompo_hd_simple/ep-7-val_loss-0.02-val_f1-0.94-train_loss-0.00-train_f1-0.98',
    processed_data_dir='data_sompo_hd',
    raw_data_dir='data_sompo_hd/test/',
    pred_data_dir='data_sompo_hd/debugs/',
    visualization_dir='visualize_sompo',
    kv_output_dir='kv_sompo',
    bert_model='cl-tohoku/bert-base-japanese',
    is_tokenized=True,
))

class Args:
    def __init__(self, args):
        self.__dict__ = args


args = Args(args)

if args.visualization_dir is None:
    args.visualization_dir = os.path.join(args.layout_model if args.bert_model is not None else args.bert_model, 
                                          'visualization')

Path(args.visualization_dir).mkdir(parents=True, exist_ok=True)
Path(args.kv_output_dir).mkdir(parents=True, exist_ok=True)

process(args)