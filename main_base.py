"""
- Fine-tune pre-trained language models (PLMs)
- SeqClsModel in Huggingface
"""
import torch
from transformers import AdamW

import numpy as np
import random

from model.train import LinearScheduler, batch_len
from model.model_adv import *
from model.load_model import *

from utils.utils import print_args, save_checkpoint, load_checkpoint, model_evaluation
from utils.dataloader import trans_dataloader 
import utils.logger as logger
import time, datetime
from datetime import timedelta

from arguments import get_parser

args = get_parser("Training") 
print_args(args)

print("Setup Logger...")
now = datetime.datetime.now()
args.exp_dir = args.exp_dir + f"{now.year}_{now.month}_{now.day}/"

if args.eval:
    print(f"Experiment Dir: {args.exp_dir} || Load Model {args.load_model}-------------------")
else:
    print(f"Experiment Dir: {args.exp_dir} || Save Model {args.save_model}-------------------")

log_path = logger.log_path(now, args.exp_dir, args.save_model)
exp_log = logger.setup_logger('perf_info', log_path)

if args.seed>0:
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

tokenizer = load_tokenizer(args)

train_dataloader, test_dataloader, dev_dataloader = \
        trans_dataloader(args.dataset, tokenizer, args)

train_niter = len(train_dataloader)
total_iter = len(train_dataloader) * args.epochs

# Create Model 
model = load_base_model(args)

if args.eval==True:
    model = load_checkpoint(model, args.load_model, args.model_dir_path)
    model.to(args.device)
    model.eval()

    TP = 0
    TP_ood = 0
    n_samples = len(test_dataloader.dataset)
    n_detection = 0

    start_t_gen = time.perf_counter()
    print("Start Evaluation....") 
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device) 
            b_length = batch_len(input_ids, args.pad_idx)

            output = model(input_ids, attention_mask)

            preds = output['logits']
            correct = preds.argmax(dim=-1).eq(labels)
            TP += correct.sum().item()

    acc = 100*(TP/n_samples)

    eval_t = time.perf_counter()-start_t_gen
    log = f"Test Acc: {acc:.4f}"
    print(log, flush=True)
    print(f"Total Evaluation Time: {timedelta(seconds=eval_t)}", flush=True) 
    exit(0)

else:
    model.to(args.device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)

print("Start Training...")
start_train = time.perf_counter()

logger.args_logger(args, args.exp_dir)

best_dev_epoch = 0
best_dev_acc = 0

data_collator = None

for epoch in range(args.epochs):
    model.train()
    loss_epoch = []
    loss_ood_epoch = []

    for batch_idx, batch in enumerate(train_dataloader):

        optimizer.zero_grad()           

        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device) 
        b_length = batch_len(input_ids, args.pad_idx)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = output['loss']
        loss.backward()    
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()   
        loss_epoch.append(loss.item())

        if batch_idx % 100 == 0:
            log = f"Epoch: {epoch} || Iter: {batch_idx} || Loss: {np.mean(loss_epoch[-100:]):.3f}"
            print(log, flush=True)
            exp_log.info(log)

        if args.scheduler=='cosine':
            lr_scheduler.step()
        elif args.scheduler=='linear':
            curr = epoch * train_niter + batch_idx
            LinearScheduler(optimizer, total_iter, curr, args.lr)

    log = f"\nEpoch: {epoch} || Loss: {np.mean(loss_epoch):.3f}"
    print(log, flush=True)
    exp_log.info(log)

    dev_acc = model_evaluation(model, dev_dataloader, args, eval_mode='dev', data_collator=data_collator)

    if dev_acc>best_dev_acc:
        best_dev_acc = dev_acc
        best_dev_epoch = epoch
        save_checkpoint(args.save_model, model, epoch, ckpt_dir=args.model_dir_path)

    log = f"Epoch: {epoch} || Dev Acc: {dev_acc:.4f} || BestDevAcc: {best_dev_acc:.4f} || BestEpoch: {best_dev_epoch}"
    exp_log.info(log)
    print(log)

end_train = time.perf_counter()-start_train
log = f"Total Training Time: {timedelta(seconds=end_train)}"
exp_log.info(log)
print(log, flush=True)

print("Start TestSet Evaluation...") 

load_model_name = args.save_model + f"_{best_dev_epoch}"
print(f"Load BestDev Model...: {load_model_name}") 

model = load_base_model(args)
model = load_checkpoint(model, load_model_name, args.model_dir_path)
model.to(args.device)
model.eval()

test_acc = model_evaluation(model, test_dataloader, args, eval_mode='test', data_collator=data_collator)
log = f"TestAcc: {test_acc:.4f} || BestDevAcc: {best_dev_acc:.4f}"
exp_log.info(log)
print(log) 
print("End Training...")

