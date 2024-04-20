from dataloader.base import AbstractDataloader
from dataloader.utils import Prompter

import torch
import random
import numpy as np
import torch.utils.data as data_utils

import os
import pickle
import transformers
from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
from trainer import absolute_recall_mrr_ndcg_for_ks


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# the following prompting is based on alpaca
def generate_and_tokenize_eval(args, data_point, tokenizer, prompter):
    in_prompt = prompter.generate_prompt(data_point["system"],
                                         data_point["input"])
    tokenized_full_prompt = tokenizer(in_prompt,
                                      truncation=True,
                                      max_length=args.llm_max_text_len,
                                      padding=False,
                                      return_tensors=None)
    tokenized_full_prompt["labels"] = ord(data_point["output"]) - ord('A')
    
    return tokenized_full_prompt


def generate_and_tokenize_train(args, data_point, tokenizer, prompter):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt,
                           truncation=True,
                           max_length=args.llm_max_text_len,
                           padding=False,
                           return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    full_prompt = prompter.generate_prompt(data_point["system"],
                                           data_point["input"],
                                           data_point["output"])
    tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
    if not args.llm_train_on_inputs:
        tokenized_full_prompt["labels"][:-2] = [-100] * len(tokenized_full_prompt["labels"][:-2])
    
    return tokenized_full_prompt


def seq_to_token_ids(args, seq, candidates, label, text_dict, tokenizer, prompter, eval=False):
    def truncate_title(title):
        title_ = tokenizer.tokenize(title)[:args.llm_max_title_len]
        title = tokenizer.convert_tokens_to_string(title_)
        return title

    seq_t = ' \n '.join(['(' + str(idx + 1) + ') ' + truncate_title(text_dict[item]) 
                    for idx, item in enumerate(seq)])
    can_t = ' \n '.join(['(' + chr(ord(char_class)) + ') ' + truncate_title(text_dict[item])
                    for item, char_class in zip(candidates, args.class_list)])
    output = chr(ord('A') + candidates.index(label))  # ranking only
    
    if args.signal == 'like':
        prompt_signal = 'liked'
    elif args.signal == 'write':
        prompt_signal = 'written'
    elif args.signal == 'both':
        prompt_signal = 'liked or written'
        
    if args.llm_system_template:
        system_template = args.llm_system_template.format(prompt_signal)
    else:
        system_template = DEFAULT_SYSTEM_PROMPT
    input_template = args.llm_input_template.format(prompt_signal, seq_t, can_t)
    
    data_point = {}
    data_point['system'] = system_template
    data_point['input'] = input_template
    data_point['output'] = output
    
    if eval:
        return generate_and_tokenize_eval(args, data_point, tokenizer, prompter)
    else:
        return generate_and_tokenize_train(args, data_point, tokenizer, prompter)


class LLMDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        seq_dataset = dataset.load_dataset()
        self.train_seq = seq_dataset['train_seq']
        self.train_cand = seq_dataset['train_cand']
        self.val_seq = seq_dataset['val_seq']
        self.val_cand = seq_dataset['val_cand']
        self.val_labels = seq_dataset['val_labels']
        self.test_seq = seq_dataset['test_seq']
        self.test_cand = seq_dataset['test_cand']
        self.test_labels = seq_dataset['test_labels']
        self.umap = seq_dataset['umap']
        self.smap = seq_dataset['smap']
        self.text_dict = seq_dataset['meta']
        self.user_count = len(self.train_seq)
        self.item_count = len(self.smap)
        
        args.num_items = self.item_count
        self.max_len = args.llm_max_history
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_base_tokenizer, cache_dir=args.llm_cache_dir)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.clean_up_tokenization_spaces = True
        self.prompter = Prompter()
        

    @classmethod
    def code(cls):
        return 'llm'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.lora_micro_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                                           worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = LLMTrainDataset(self.args, self.train_seq, self.train_cand, self.max_len, self.rng,
                                  self.text_dict, self.tokenizer, self.prompter)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LLMValidDataset(self.args, self.val_seq, self.val_cand, self.val_labels, self.max_len, \
                                      self.rng, self.text_dict, self.tokenizer, self.prompter, self.save_folder)
        elif mode == 'test':
            dataset = LLMTestDataset(self.args, self.test_seq, self.test_cand, self.test_labels, self.max_len, \
                                     self.rng, self.text_dict, self.tokenizer, self.prompter, self.save_folder)
        return dataset


class LLMTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2cand, max_len, rng, text_dict, tokenizer, prompter):
        self.args = args
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter

        self.all_seqs = []
        self.all_answers = []
        self.all_cands = []
        for seq, cand in zip(u2seq, u2cand):
            self.rng.shuffle(cand)
                
            self.all_seqs += [seq]
            self.all_cands += [cand]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        tokens = self.all_seqs[index]
        answer = tokens[-1]
        original_seq = tokens[:-1]
        
        seq = original_seq[-self.max_len:]
        candidates = [answer]
        neg_samples = self.all_cands[index]
        candidates.extend(neg_samples)
        self.rng.shuffle(candidates)

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, \
                                self.tokenizer, self.prompter, eval=False)


class LLMValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2cand, u2labels, max_len, rng, text_dict, tokenizer, prompter, save_folder):
        self.args = args
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        
        self.all_seqs = []
        self.all_answers = []
        self.all_cands = []
        self.all_labels = []
        for seq, cand, labels in zip(u2seq, u2cand, u2labels):
            for i in range(args.llm_bootstrap):
                self.rng.shuffle(cand)
                self.all_labels.append([labels[i] for i in cand])
                    
                self.all_seqs += [seq]
                self.all_cands += [cand]
                self.all_answers.append([c for c in cand if labels[c] >= 3])
        
        with open(os.path.join(args.export_root, 'valid_labels.pkl'), 'wb') as f:
            pickle.dump(self.all_labels, f)

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        
        seq = seq[-self.max_len:]
        candidates = self.all_cands[index]
        possible_answers = self.all_answers[index]
        if possible_answers:
            answer = random.choice(possible_answers)
        else:
            answer = random.choice(candidates)
        
        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, self.tokenizer, self.prompter, eval=True)


class LLMTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2cand, u2labels, max_len, rng, text_dict, tokenizer, prompter, save_folder):
        self.args = args
        self.u2seq = u2seq
        self.max_len = max_len
        self.rng = rng
        self.text_dict = text_dict
        self.tokenizer = tokenizer
        self.prompter = prompter
        
        self.all_seqs = []
        self.all_answers = []
        self.all_cands = []
        self.all_labels = []
        for seq, cand, labels in zip(u2seq, u2cand, u2labels):
            for i in range(args.llm_bootstrap):
                self.rng.shuffle(cand)
                self.all_labels.append([labels[i] for i in cand])
                    
                self.all_seqs += [seq]
                self.all_cands += [cand]
                self.all_answers.append([c for c in cand if labels[c] >= 3])
        
        with open(os.path.join(args.export_root, 'test_labels.pkl'), 'wb') as f:
            pickle.dump(self.all_labels, f)
    
    def __len__(self):
        return len(self.all_seqs)
    
    def __getitem__(self, index):
        seq = self.all_seqs[index]
        
        seq = seq[-self.max_len:]
        candidates = self.all_cands[index]
        possible_answers = self.all_answers[index]
        if possible_answers:
            answer = random.choice(possible_answers)
        else:
            answer = random.choice(candidates)

        return seq_to_token_ids(self.args, seq, candidates, answer, self.text_dict, self.tokenizer, self.prompter, eval=True)
