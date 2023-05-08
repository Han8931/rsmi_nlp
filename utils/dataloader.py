import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, list_datasets

import numpy as np
import pandas as pd
import pdb
import string, operator
import re

from abc import *
import json

import random
import sys, os, pdb
import time, datetime
import argparse, copy

import string

class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def trans_dataloader(dataset, tokenizer, args, data_collator=None):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        split = train.train_test_split(test_size=0.10, seed=0)
        train = split['train']
        dev = split['test']

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        split = train.train_test_split(test_size=0.10, seed=0)
        train = split['train']
        dev = split['test']

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")
    print(f"Devset Size: {len(dev)}")

    train_data = tokenizer(train['text'], padding=True, truncation=True, max_length=args.max_seq_length)
    test_data = tokenizer(test['text'], padding=True, truncation=True, max_length=args.max_seq_length)
    dev_data = tokenizer(dev['text'], padding=True, truncation=True, max_length=args.max_seq_length)
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = train['label']
    test_label = test['label']
    dev_label = dev['label']

    if args.self_supervised:
        train_set = SSLDataset(train_data, train_label)
        test_set = SSLDataset(test_data, test_label)
    else:
        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)
        dev_set = ClsDataset(dev_data, dev_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

    return train_dataloader, test_dataloader, dev_dataloader


def text_dataloader(dataset, args):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        split = train.train_test_split(test_size=0.10, seed=0)
        train = split['train']
        dev = split['test']

        args.num_classes = 4

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        split = train.train_test_split(test_size=0.10, seed=0)
        train = split['train']
        dev = split['test']

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")


    return train, test

