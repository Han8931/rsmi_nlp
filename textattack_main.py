"""
Adversarial example generation scripts with TextAttack_v2
"""
import torch
from torch.utils.data import DataLoader

import pdb
import numpy as np
import random
import os, sys
import argparse
import pandas as pd


from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
from model.textattack_model import CustomWrapper, print_function, AttackSummary, HuggingFaceModelWrapper

from model.model_adv import *
from model.load_model import *

from utils.utils import boolean_string, print_args, load_checkpoint, 
from utils.dataloader import text_dataloader

from datetime import timedelta
import time, datetime

from arguments import get_parser

import warnings
warnings.filterwarnings('ignore')

args = get_parser("attack") 

if args.seed>-1:
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

max_pert_ratio = int(args.max_rate*100) 

n_ens = args.num_ensemble
egm = args.ens_grad_mask
gms = args.grad_mask_sample
mpr = max_pert_ratio
ql = args.q_limit

if args.nth_data == 0:
    f_name = "_"+str(args.load_model)+"_"+args.dataset_type+f"_ens_{n_ens}_{mpr}_egm_{egm}_gs_{gms}_nq_{ql}_ts_{args.two_step}.csv"
else: 
    f_name = "_"+str(args.load_model)+"_"+args.dataset_type+f"_ens_{n_ens}_{mpr}_egm_{egm}_gs_{gms}_nq_{ql}_ts_{args.two_step}_{args.nth_data}.csv"
adv_path = os.path.join('./data/'+args.dataset+'_'+args.attack_method+f_name)

args.adv_path = adv_path

print("Load Dataset...") 

if args.dataset_type =='test':
    _, test = text_dataloader(args.dataset, args)
    dataset = test.train_test_split(test_size=args.n_trials, seed=0)['test']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=args.shuffle, num_workers=0)
else:
    train, _ = text_dataloader(args.dataset, args)
    dataset = train.train_test_split(test_size=args.n_trials, seed=0)['test']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=args.shuffle, num_workers=0)

print_args(args)
print(f"Adv Path: {adv_path}") 
print(f"Save Data: {args.save_data}") 

print("----------------------------------")
print(f"Dataset Type: {args.dataset_type}") 
print("----------------------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Model 
print(f"Load Model...")

if args.model == 'bert':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'roberta':
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

args.pad_idx = tokenizer.pad_token_id
args.mask_idx = tokenizer.mask_token_id
args.cls_token = tokenizer.cls_token_id
args.sep_token = tokenizer.sep_token_id

if args.model_type=='rsmi':
    print("RSMI is Loaded")
    model = noisy_forward_loader(args)
    model = SeqClsWrapper(model, args)

elif args.model_type=='base':
    print("Base Model is Loaded")
    model = load_base_model(args)

model = load_checkpoint(model, args.load_model, args.model_dir_path)
model.eval()

if args.model_type=='base':
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer, args)
else:
    print("Custom Wrapper is used...")
    model_wrapper = CustomWrapper(model, tokenizer, args)
model_wrapper.model.to(device)

# Attack Recipe
if args.attack_method == 'pwws':
    from textattack.attack_recipes import PWWSRen2019
    attack = PWWSRen2019.build(model_wrapper)
elif args.attack_method == 'textfooler':
    from textattack.attack_recipes import TextFoolerJin2019
    attack = TextFoolerJin2019.build(model_wrapper)

attack.goal_function.maximizable = False
attack.goal_function.batch_size = args.adv_batch_size
print(attack)

n_trial = 0
num_successes = 0
num_skipped = 0
num_failed = 0

total_queries = 0

df_adv = pd.DataFrame()
attack_result = AttackSummary(args.max_seq_length)

n_exception = 0
n_pred_error = 0

avg_pert = []
avg_query = []
avg_words = []

start_t_gen = time.perf_counter()

print("Start Attack...")
for batch_idx, batch in enumerate(dataloader):

    label = batch['label'].item() # Ground truth label
    orig = batch['text'][0] # Clen Text 

    text_len = len(orig.split(" "))

    if text_len>args.max_seq_length:
        orig = " ".join(orig.split(" ")[:256])
        text_len = args.max_seq_length

    if args.q_limit>0:
        q_limit = int(args.max_candidates*text_len)
        attack.goal_function.query_budget = q_limit

    result = attack.attack(orig, label)

    attack_result(result)
    n_query, n_pert, n_words = attack_result.text_analysis(result)

    pert = result.perturbed_text()

    pert_word_ratio =(n_pert/text_len)*100

    if isinstance(result, SuccessfulAttackResult):
        num_successes+=1
        result_type = 'Successful'
    elif isinstance(result, FailedAttackResult):
        num_failed+=1
        result_type = 'Failed'
    elif isinstance(result, SkippedAttackResult):
        num_skipped+=1
        result_type = 'Skipped'

    adv_dict = {'pert': pert, 'orig': orig, 'ground_truth_output': label, 
            'result_type': result_type, 'n_query': n_query, 'n_pert': n_pert, 'n_words': n_words}
    df_adv = df_adv.append(adv_dict, ignore_index=True)

    if batch_idx%10==0:
        print(result.__str__(color_method='ansi'))
        print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped)
        print(attack_result.__str__(), flush=True)

eval_t = time.perf_counter()-start_t_gen
print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped)
print(attack_result.__str__(), flush=True)
print(f"Total Elapsed Time: {timedelta(seconds=eval_t)}", flush=True) 

if args.save_data == True:
    if not os.path.isdir('./data/'):
        os.makedirs('./data/')
    
    df_adv.to_csv(adv_path)
    print(f"Save data...: {adv_path}", flush=True) 


