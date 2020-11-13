import os

import torch
from torch import nn
from tqdm import tqdm, trange
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import DatasetForTokenClassification
from utils import *

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


args = dict(
    data_dir='sroie_multiline_SO_with_val', 
    max_seq_length=512,
    eval_batch_size=16,
    model_name_or_path='test_sroie_SO_2', 
    model_type='layoutlm', 
    device='cuda',
    eval_all_checkpoints=True,
    overwrite_cache=True,
    smoothened=False,
    so_only=True
)
class Args:
    def __init__(self, args):
        self.__dict__ = args

args = Args(args)


pad_token_label_id = nn.CrossEntropyLoss().ignore_index

tokenizer = LayoutLMTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=True)

model = LayoutLMForTokenClassification.from_pretrained(args.model_name_or_path, 
                                                       return_dict=True)

model.to(args.device)

labels = get_labels(os.path.join(args.data_dir, 'labels.txt'))
label_map = {i: label for i, label in enumerate(labels)}
eval_dataset = DatasetForTokenClassification(args, tokenizer, labels, pad_token_label_id, mode="val")

eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(
    eval_dataset,
    sampler=eval_sampler,
    batch_size=args.eval_batch_size,
    collate_fn=None,
)

print("  Num examples =", len(eval_dataset))

preds = None
out_label_ids = None
model.eval()
for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
    with torch.no_grad():
        inputs = {
            "input_ids": batch[0].to(args.device),
            "attention_mask": batch[1].to(args.device),
            "token_type_ids": batch[2].to(args.device),
            # "labels": batch[3].to(args.device),
            "bbox": batch[4].to(args.device)
        }
        
        outputs = model(**inputs)
        logits = outputs.logits
        
    p = logits.detach().cpu().numpy()
    l = batch[3].detach().cpu().numpy()
    
    
    if preds is None:
        preds = p
        out_label_ids = l
    else:
        preds = np.append(preds, p, axis=0)
        out_label_ids = np.append(out_label_ids, l, axis=0)

preds = np.argmax(preds, axis=2)
if args.smoothened:
    preds = [smoothen(p) for p in preds]

out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != pad_token_label_id:
            out_label_list[i].append(label_map[out_label_ids[i][j]])
            preds_list[i].append(label_map[preds[i][j]])

results = {
    "precision": precision_score(out_label_list, preds_list),
    "recall": recall_score(out_label_list, preds_list),
    "f1": f1_score(out_label_list, preds_list),
}

if args.so_only:
    for i, p in enumerate(preds_list):
        preds_list[i] = convert_SO_to_BIOES(p)
    for i, l in enumerate(out_label_list):
        out_label_list[i] = convert_SO_to_BIOES(l)
    BIOES_results = {
        "BIOES_precision": precision_score(out_label_list, preds_list),
        "BIOES_recall": recall_score(out_label_list, preds_list),
        "BIOES_f1": f1_score(out_label_list, preds_list),
    }
    results.update(BIOES_results)

report = classification_report(out_label_list, preds_list)
print("\n", report)

print("***** Eval results*****")
for key in sorted(results.keys()):
    print("  %s = %s" % (key, str(results[key])))

# args.data_dir = 'sroie_multiline_SO_with_val'
# labels = get_labels(os.path.join(args.data_dir, 'labels.txt'))
# label_map2 = {i: label for i, label in enumerate(labels)}
# eval_dataset2 = DatasetForTokenClassification(args, tokenizer, labels, pad_token_label_id, mode="val")
# 
# eval_sampler2 = SequentialSampler(eval_dataset2)
# eval_dataloader2 = DataLoader(
#     eval_dataset2,
#     sampler=eval_sampler2,
#     batch_size=args.eval_batch_size,
#     collate_fn=None,
# )
# 
# print("  Num examples =", len(eval_dataset2))

# cnt = 1
# for u, v in zip(eval_dataloader, eval_dataloader2):
#     for ub, vb in zip(u[3], v[3]):
#         full = [label_map[i] for i in ub.detach().cpu().numpy() if i != -100]
#         l = [i for i in vb.detach().cpu().numpy() if i != -100]
#         # l = smoothen(l)
#         so = [label_map2[i] for i in l if i != -100]
#         bioes = convert_SO_to_BIOES(so)
#         print(cnt, full == bioes)
#         cnt += 1

# a = list(eval_dataloader)[0]
# full = [label_map[i] for i in a[3][0].detach().cpu().numpy() if i != -100]
# 
# b = list(eval_dataloader2)[0]
# l = [i for i in b[3][0].detach().cpu().numpy() if i != -100]
# # l = smoothen(l)
# so = [label_map2[i] for i in l if i != -100]
# bioes = convert_SO_to_BIOES(so)

