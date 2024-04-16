from config import *
from .verb import ManualVerbalizer
from .utils import *
from .loggers import *
from .base import *

import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
from abc import *
from pathlib import Path

import bitsandbytes as bnb
from transformers.trainer import *
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def llama_collate_fn_w_truncation(llm_max_length, eval=False):
    def llama_collate_fn(batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        example_max_length = max([len(batch[idx]['input_ids']) for idx in range(len(batch))])
        max_length = min(llm_max_length, example_max_length)
        
        for i in range(len(batch)):
            input_ids = batch[i]['input_ids']
            attention_mask = batch[i]['attention_mask']
            labels = batch[i]['labels']
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                if not eval: labels = labels[-max_length:]
            elif len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids = [0] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                if not eval: labels = [-100] * padding_length + labels

            if eval: assert input_ids[-1] == 13
            else:
                assert input_ids[-3] == 13 and input_ids[-1] == 2
                assert labels[-3] == -100 and labels[-2] != -100
            
            all_input_ids.append(torch.tensor(input_ids).long())
            all_attention_mask.append(torch.tensor(attention_mask).long())
            all_labels.append(torch.tensor(labels).long())
        
        return {
            'input_ids': torch.vstack(all_input_ids),
            'attention_mask': torch.vstack(all_attention_mask),
            'labels': torch.vstack(all_labels)
        }
    return llama_collate_fn


def compute_metrics_for_ks(ks, verbalizer):
    def compute_metrics(eval_pred, prefix=''):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).view(-1)
        scores = verbalizer.process_logits(logits)
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, ks, prefix)
        return metrics
    return compute_metrics


class LLMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            tokenizer,
            export_root,
            use_wandb,
            **kwargs
        ):
        self.original_args = args
        self.export_root = export_root
        self.use_wandb = use_wandb
        self.llm_max_text_len = args.llm_max_text_len
        self.rerank_metric_ks = args.rerank_metric_ks
        self.verbalizer = ManualVerbalizer(
            tokenizer=tokenizer,
            prefix='',
            post_log_softmax=False,
            classes=list(range(args.llm_negative_sample_size+1)),
            label_words={i: chr(ord(char_class)) for i, char_class in enumerate(args.class_list)}
        )
        
        wandb_run_name = f'LlamaRec_{args.dataset_code}{args.category.replace(" & ", "_")}_{args.signal}{args.test}'
        if args.summary:
            wandb_run_name = wandb_run_name + '_summary'

        hf_args = TrainingArguments(
            per_device_train_batch_size=args.lora_micro_batch_size,
            per_device_eval_batch_size=args.lora_micro_batch_size,
            gradient_accumulation_steps=args.train_batch_size//args.lora_micro_batch_size,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.lora_num_epochs,
            learning_rate=args.lora_lr,
            bf16=True,
            logging_steps=10,
            optim="paged_adamw_32bit",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.lora_val_iterations,
            save_steps=args.lora_val_iterations,
            output_dir=export_root,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            metric_for_best_model=args.rerank_best_metric,
            greater_is_better=True,
        )
        super().__init__(
            model=model,
            args=hf_args,
            callbacks=[EarlyStoppingCallback(args.lora_early_stopping_patience)],
            **kwargs)  # hf_args is now args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        
        self.train_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=False)
        self.val_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=True)
        self.test_loader.collate_fn = llama_collate_fn_w_truncation(self.llm_max_text_len, eval=True)
        self.compute_metrics = compute_metrics_for_ks(self.rerank_metric_ks, self.verbalizer)

        if len(self.label_names) == 0:
            self.label_names = ['labels']  # for some reason label name is not set
    
    def test(self):
        if args.summary:
            model_name = f'{args.dataset_code}{args.category}_{args.signal}_summary'
        else:
            model_name = f'{args.dataset_code}{args.category}_{args.signal}'
        average_metrics = self.predict(test_dataset=None).metrics
        average_metrics['model_name'] = model_name
        average_metrics['learning_rate'] = args.lora_lr
        print('Ranking Performance on test:', average_metrics)
        print('************************************************************')
        with open(os.path.join(EXPERIMENT_ROOT, 'model_results.json'), 'a+') as f:
                json.dump(average_metrics, f, indent=4)
        
        return average_metrics

    def get_train_dataloader(self):
        return self.train_loader
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.val_loader
    
    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.test_loader
