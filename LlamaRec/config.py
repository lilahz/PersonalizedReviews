import numpy as np
import random
import torch
import argparse
import string


RAW_DATASET_ROOT_FOLDER = '/sise/bshapira-group/lilachzi/models/LlamaRec/data'
EXPERIMENT_ROOT = '/sise/bshapira-group/lilachzi/models/LlamaRec/experiments/'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'llmrec'


def set_template(args):
    print(f'DEBUG: {args.dataset_code}{args.category} - {args.signal}')
    args.test = '_8B_10_bs'
    print(f'DEBUG: {args.test}')

    batch = 8
    args.lora_micro_batch_size = batch

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch
    
    args.llm_max_history = 15

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.0001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 10

    args.metric_ks = [1, 3, 5, 50]
    args.rerank_metric_ks = [1, 3, 5]
    args.best_metric = 'Recall@5'
    args.rerank_best_metric = 'nDCG@5'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None
    
    args.class_list = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    args.class_list = args.class_list[:(args.llm_negative_sample_size + 1)]


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='ciao')
parser.add_argument('--category', type=str, default='Beauty')
parser.add_argument('--signal', type=str, default='like')
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--export_root', type=str, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=5)
parser.add_argument('--min_sc', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)

################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--sliding_window_size', type=int, default=10)
parser.add_argument('--sliding_window_step', type=int, default=5)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=10)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=20)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=2)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 3, 5])
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 3, 5])
parser.add_argument('--best_metric', type=str, default='Recall@5')
parser.add_argument('--rerank_best_metric', type=str, default='nDCG@5')
parser.add_argument('--use_wandb', type=bool, default=True)

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.2)
parser.add_argument('--bert_attn_dropout', type=float, default=0.2)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# LLM Model
################
parser.add_argument('--llm_base_model', type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument('--llm_base_tokenizer', type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument('--llm_max_title_len', type=int, default=50)
parser.add_argument('--llm_max_text_len', type=int, default=2500)
parser.add_argument('--llm_max_history', type=int, default=20)
parser.add_argument('--llm_train_on_inputs', type=bool, default=False)
parser.add_argument('--llm_negative_sample_size', type=int, default=9)  # 19 negative & 1 positive
parser.add_argument('--llm_bootstrap', type=int, default=3)
parser.add_argument('--llm_system_template', type=str,  # instruction
    default="Given user history of reviews they previously {}, recommend the reviews the user will like the most from the candidate pool using the reviews index letters.")
parser.add_argument('--llm_input_template', type=str, \
    default='User {} history: {}; \n Candidate pool: {}')
parser.add_argument('--llm_load_in_4bit', type=bool, default=True)
parser.add_argument('--llm_retrieved_path', type=str, default=None)
parser.add_argument('--llm_cache_dir', type=str, default=None)

################
# Lora
################
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=64)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'v_proj'])
parser.add_argument('--lora_num_epochs', type=int, default=1)
parser.add_argument('--lora_val_iterations', type=int, default=50)
parser.add_argument('--lora_early_stopping_patience', type=int, default=2)
parser.add_argument('--lora_lr', type=float, default=1e-3)
parser.add_argument('--lora_micro_batch_size', type=int, default=16)

################


args = parser.parse_args()
