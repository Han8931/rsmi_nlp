# Randomized Smoothing with Masked Inference for Adversarially Robust NLP Systems
MIT License

Copyright (c) [2022] [Anonymous]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Train Models:

### Fine-tune PLMs
```
CUDA_VISIBLE_DEVICES=0 python3 main_base.py --model_dir_path ./cls_task/checkpoint/ --dataset ag --batch_size 24 --epochs 10 --save_model base_roberta_ag --model roberta --save --max_seq_length 256 --lr 0.00001
```

### RSMI
```
CUDA_VISIBLE_DEVICES=0 python3 main_org_seq.py --model_dir_path ./cls_task/checkpoint/ --dataset imdb --batch_size 16 --epochs 10 --save_model test --model roberta --nth_layers 3 --noise_eps 0.2 --max_seq_length 256 --multi_mask 2
```

## Attack Models:

### Fine-tuned PLMs
```
CUDA_VISIBLE_DEVICES=0 python3 textattack_main.py --model_dir_path ./cls_task/checkpoint/ --load_model base_roberta_ag_0 --dataset imdb --nth_data 0 --seed 0 --dataset_type test --save_data --model roberta --attack_method textfooler --n_success 1000 --batch_size 1 --max_seq_length 256 --model_type base --max_rate 1.0
```

### RSMI
```
CUDA_VISIBLE_DEVICES=0 python3 textattack_main.py --model_dir_path ./cls_task/checkpoint/ --load_model base_roberta_ag --dataset ag --nth_data 0 --seed 0 --dataset_type test --save_data True --model roberta --attack_method textfooler --n_success 1000 --batch_size 1 --nth_layer 3 --noise_eps 0.3 --multi_mask 3 --max_rate 1.0 --num_ensemble 5 --custom_forward True --hf_model True --max_seq_length 256 --model_type  --exp_msg Test --adv_batch_size 80 --alpha_p 0.98 --ens_grad_mask rand --two_step True
```
