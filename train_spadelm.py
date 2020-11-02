import os
import json
import time
import logging
import datetime
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from imutils import paths
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    LayoutLMTokenizer, 
    LayoutLMForTokenClassification,
    LayoutLMModel, 
    AdamW,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from funsd import FunsdDataset

from spade.spadelm import SpadeLMForTokenClassification, SpadeLMConfig

logger = logging.getLogger(__name__)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="",
             verbose=True):
    eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode=mode)
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )
    
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not verbose):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": batch[2].to(args.device),
                "labels": batch[3].to(args.device),
                "bbox": batch[4].to(args.device)
            }
            
            outputs = model(**inputs)
            tmp_eval_loss = outputs.loss
            logits = outputs.logits
            
            eval_loss += tmp_eval_loss.item()
        
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
    
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)
    
    label_map = {i: label for i, label in enumerate(labels)}
    
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    
    report = classification_report(out_label_list, preds_list)
    logger.info("\n" + report)
    
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results, preds_list


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    summary_writer = SummaryWriter(logdir="runs/" + os.path.basename(args.output_dir))

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=None,
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(args.num_train_epochs, desc="Epoch")

    best_loss = 999
    for i in train_iterator:
        print()
        print('Epoch', i, '=' * 17)
        epoch_iterator = tqdm(train_dataloader, desc='Epochs {}'.format(i))
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            inputs = {
                'input_ids': batch[0].to('cuda'),
                'attention_mask': batch[1].to('cuda'),
                'token_type_ids': batch[2].to('cuda'),
                'labels': batch[3].to('cuda'),
                'bbox': batch[4].to('cuda')
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    summary_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    summary_writer.add_scalar(
                        "train_loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss
                
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        train_results, _ = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="train",
            verbose=False
        )
        print('train:', train_results)
        val_results, _ = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="val",
            verbose=False
        )
        print('val:', val_results)
        for key, value in val_results.items():
            summary_writer.add_scalar(
                "val_{}".format(key), value, global_step
            )
        if val_results['loss'] < best_loss:
            best_loss = val_results['loss']
            print('saving best loss', best_loss)
            # Save model checkpoint
            output_dir = os.path.join(
                args.output_dir, 
                """ep-{}-val_loss-{:.2f}-val_f1-{:.2f}-train_loss-{:.2f}
                -train_f1-{:.2f}""".replace(' ', '').format(i, best_loss, 
                                                           val_results['f1'], 
                                                           train_results['loss'], 
                                                           train_results['f1'])
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
        
        print()
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
    summary_writer.close()
    
    return global_step, tr_loss / global_step


args = dict(
    data_dir='data_with_val', 
    labels='data_with_val/labels.txt', 
    max_seq_length=512, 
    model_name_or_path='layoutlm-base-uncased', 
    model_type='spadelm', 
    num_train_epochs=2, 
    learning_rate=5e-5,
    weight_decay=0.0,
    output_dir='test_spade', 
    overwrite_cache=False, 
    train_batch_size=1,
    eval_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    adam_epsilon=1e-8,
    max_steps=-1,
    save_steps=-1, 
    logging_steps=1,
    max_grad_norm=1.0, 
    device='cuda'
)


class Args:
    def __init__(self, args):
        self.__dict__ = args

args = Args(args)

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(args.output_dir, "train.log"),
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


labels = get_labels(args.labels)
num_labels = len(labels)

pad_token_label_id = nn.CrossEntropyLoss().ignore_index


tokenizer = LayoutLMTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=True)

config = SpadeLMConfig(**{
            "hidden_size": 768, 
            "vocab_size": 30522, 
            "pad_token_id": 0, 
            'layer_norm_eps': 1e-12, 
            'hidden_dropout_prob': 0.1,
            'batch_size': args.train_batch_size, 
            'sequence_len': args.max_seq_length, 
            'num_attention_heads': 12, 
            'intermediate_size': 3072, 
            'n_layers': 12,
            'num_labels': num_labels,
            'return_dict': True
        })


model = SpadeLMForTokenClassification(config)

# self_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 8, 2048, 0.2), 1)
# # from custom_attention import CustomAttentionLayer
# # self_att = nn.TransformerEncoder(CustomAttentionLayer(768, 8, 2048, 0.0), 2)
# fc = nn.Linear(768, num_labels)
# model.classifier = nn.Sequential(self_att, fc)

model.to('cuda')

for param in model.base_model.parameters():
    param.requires_grad = False

logger.info("Training/evaluation parameters %s", args)

logger.info('Training...')

train_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")

global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)

# Create output directory if needed
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger.info("Saving model checkpoint to %s", args.output_dir)
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Good practice: save your training arguments together with the trained model
torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


logger.info('Evaluating...')

results = {}
result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, 
                     mode="test", prefix=global_step)
if global_step:
    result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
    results.update(result)

output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    for key in sorted(results.keys()):
        writer.write("{} = {}\n".format(key, str(results[key])))

