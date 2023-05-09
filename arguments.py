import torch
import argparse


def get_parser(process_type: str):
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--exp_dir', type=str, default="./experiments/cls/", help='Experiment directory.')
    parser.add_argument('--exp_msg', type=str, default="CLS Transformer", help='Simple log for experiment')
    parser.add_argument('--gpu_idx', type=int, default=10, help='GPU Index')

    parser.add_argument('--eval', action="store_true", help='Evaluation')
    parser.add_argument('--model_dir_path', default='./', type=str, help='Save Model dir')
    parser.add_argument('--save_model', default='cls_trans', type=str, help='Save Model name')
    parser.add_argument('--load_model', default='cls_trans', type=str, help='Model name')
    parser.add_argument('--save', action="store_true", help='Evaluation')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    # Training
    parser.add_argument('--w_gn', default=1.0, type=float, help='Grad noise weight to word embedding')
    parser.add_argument('--noise_eps', type=float, default=0.0, help='Noise Size. Stddev of Gaussian')
    parser.add_argument('--single_layer', action="store_true", help='Single layer perturbation')
    parser.add_argument('--nth_layers', type=int, default=-1, \
            help='Noise Layer insertion index. Every i-th layer. Smaller number puts more noise layers')

    # Inference
    parser.add_argument('--num_ensemble', type=int, default=1, help='Number of ensemble inputs')
    parser.add_argument('--binom_ensemble', type=int, default=50, help='Number of ensemble inputs for re-eval')
    parser.add_argument('--pooler_output', action="store_true", help='Pooler Output')
    parser.add_argument('--custom_forward', action="store_true", help='Customized forward function')
    parser.add_argument('--binom_n_eval', type=int, default=5, help='Number of re-eval')


    parser.add_argument('--sample_mask', action="store_true", help='Sample a few mask indices from grad mask indices')
    parser.add_argument('--mask_batch_ratio', type=float, default=1.0, help='Sample and Mask bach ratio')
    parser.add_argument('--rand_mask', action="store_true", help='Masking input randomly')
    parser.add_argument('--grad_mask_sample', action="store_true", help='Sampling based GradMasking')
    parser.add_argument('--ens_grad_mask', default='rand', type=str, help='EnsembleGradMaskingType', choices=['grad', 'rand', 'noise'])
    parser.add_argument('--two_step', action="store_true", help='Two-Step Monte-Carlo Sampling Ensemble')
    parser.add_argument('--multi_mask', type=int, default=0, help='Masking multiple token')
    parser.add_argument('--mask_idx', type=int, default=103, help='[MASK] idx')
    parser.add_argument('--vote_type', default='avg', type=str, help='Max/Avg voting', choices=['max', 'avg'])

    parser.add_argument('--epochs', type=int, default=10, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', default='roberta', type=str, help='Transformer model',
                        choices=['bert', 'roberta', 'roberta-large', 'bert-large'])
    parser.add_argument('--optim', default='adamw', type=str, help='optimizer')
    parser.add_argument('--scheduler', default='linear', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.00001, type=float, help='Agent learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--clip', default=1.0, type=float, help='Clip grad')
    parser.add_argument('--margin', default=0.5, type=float, help='Triplet loss margin')

    parser.add_argument('--embed_dim', type=int, default=768, help='LSTM Input_Dim')

    # Dataset
    parser.add_argument('--dataset', default='imdb', type=str, help='Dataset', choices=['ag', 'imdb'])
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes of datasets')
    parser.add_argument('--pad_idx', type=int, default=0, help='Padding idx')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')

    parser.add_argument('--binom_p', action="store_true", help='Ensemble with high-confident samples')
    parser.add_argument('--alpha_p', default=0.70, type=float, help='Statistical significance')

    if 'attack' in process_type:
        parser.add_argument('--hf_model', action="store_true", help='Huggingface Model')
        parser.add_argument('--attack_method', default='textfooler', type=str, help='TextAttack Method',
                            choices=['pwws', 'textfooler'])

        parser.add_argument('--p_prune', default=0.1, type=float, help='Embedding pruning percentage')
        parser.add_argument('--max_rate', default=1.0, type=float, help='Max Perturbation Ratio, IMDb:0.3, AG:0.1')
        parser.add_argument('--max_candidates', type=int, default=50, help='Max # Syn Candidate')

        parser.add_argument('--adv_batch_size', type=int, default=16, help='Dataset idx')

        # Dataset
        parser.add_argument('--save_data', action="store_true", help='Save data')
        parser.add_argument('--dataset_type', default='test', type=str, help='Dataset', choices=['train', 'test'])
        parser.add_argument('--nth_data', type=int, default=0, help='Dataset idx')
        parser.add_argument('--result_check', action="store_true", help='Check result one more time')

        parser.add_argument('--n_trials', type=int, default=1000, help='Num Adv Success')
        parser.add_argument('--shuffle', action="store_true", help='Dataset Split')
        parser.add_argument('--q_limit', type=int, default=0, help='Num Queries')

        parser.add_argument('--model_type', default='base', type=str, help='Model Type (base or RSMI)')
        parser.add_argument('--model_analysis', action="store_true", help='Save data')

    args = parser.parse_known_args()[0]

    return args

