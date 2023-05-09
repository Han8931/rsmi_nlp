import torch
import torch.nn as nn
import time, datetime
from datetime import timedelta
import os

from functools import wraps
import numpy as np
import pickle

def eval_time_wrapper(func):
    @wraps(func)
    def function_wrapper(model, dataloader, args, eval_mode, data_collator=None):
        start_t = time.perf_counter()
        acc = func(model, dataloader, args, eval_mode, data_collator=None)
        end_t = time.perf_counter()-start_t
        log = f"Eval: {eval_mode} || Elapsed Time: {timedelta(seconds=end_t)}"
        print(log, flush=True)
        return acc

    return function_wrapper

@eval_time_wrapper
def model_evaluation(model, dataloader, args, eval_mode=None, data_collator=None):
    model.eval()
    TP = 0
    n_samples = len(dataloader.dataset)

    if args.multi_mask>0:
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device) 

            indices, _ = model.grad_mask(input_ids, attention_mask, topk=3, pred=None, mask_filter=True)
            model.zero_grad()           

            masked_ids = input_ids.clone()
            for ids_, m_idx in zip(masked_ids, indices): # for each sample in a batch
                for j in range(args.multi_mask):
                    try:
                        ids_[m_idx[j]] = args.mask_idx
                    except:
                        continue

            with torch.no_grad():
                output = model(input_ids=masked_ids, attention_mask=attention_mask)

            preds = output['logits']
            correct = preds.argmax(dim=-1).eq(labels)
            TP += correct.sum().item()

    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):

                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device) 

                output = model(input_ids, attention_mask)
                preds = output['logits']
                correct = preds.argmax(dim=-1).eq(labels)
                TP += correct.sum().item()

    acc = 100*(TP/n_samples)

    return acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def max_min_scaler(pred):
    min_pred = pred.min()
    max_pred = pred.max()
    pred_norm = (pred-min_pred)/(max_pred-min_pred)
    return pred_norm

def save_dataset(dataset, directory, dataset_name):
    path = os.path.join(directory, dataset_name+'.pickle')
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(path, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(dataset, directory, dataset_name):
    path = os.path.join(directory, dataset_name)
    with open(path, 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset

def save_checkpoint(save_model, model, epoch, ckpt_dir):
    ckpt_name = save_model + f"_{epoch}"
    print(f"Save: {ckpt_name}")
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))


def load_checkpoint(model, model_name, ckpt_dir):
    print(f"Load: {model_name}") 
    load_path = os.path.join(ckpt_dir, model_name)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    return model

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def print_args(args):
    """
    Print all arguments in argparser
    """

    if args.device == "cuda":
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)

    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    now = datetime.datetime.now()
    print(f"||Experiment Date:{now.year}-{now.month}-{now.day}||")

    print("Arguments List: \n")
    if args.device == "cuda":
        print(f"- Running on GPU: {gpu_name} || GPU Idx: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print(f"- Running on CPU")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")

    print("---------------------------------------------------------")
    print("---------------------------------------------------------\n")



