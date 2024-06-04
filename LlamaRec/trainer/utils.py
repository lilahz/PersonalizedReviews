from config import *

import json
import math
import os
import pandas as pd
import pickle
import pprint as pp
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim as optim


def calc_ndcg(relevance_list):
    dcg = 0
    for i, relevance in enumerate(relevance_list):
        dcg += (relevance) / math.log2(i + 1 + 1)
    idcg = 0
    ideal_rank = sorted(relevance_list, reverse=True)
    for i, relevance in enumerate(ideal_rank):
        idcg += (relevance) / math.log2(i + 1 + 1)
    if idcg > 0:
        ndcg = float(dcg) / idcg
    else:
        ndcg = 0
    return ndcg

def calc_mrr(y_pred):
    rr = [1/idx for idx, y in enumerate(y_pred, start=1) if y >= 3]
    
    if not rr:
        return 0.0
    
    return sum(rr) / len(rr)

def calc_recall_at_k(y_true, y_pred, k):
    relevant = len([y for y in y_true if y >= 3])
    if y_pred is None or not relevant:
        return None
    predicted = len([y for y in y_pred[:k] if y >= 3])

    return float(predicted) / float(relevant)

def calc_precision_at_k(y_pred, k):
    predicted = len([y for y in y_pred[:k] if y >= 3])

    return float(predicted) / float(k)

def calc_hit_at_k(y_pred, k):
    predicted = len([y for y in y_pred[:k] if y >= 3])
    
    if predicted > 0:
        return 1
    else:
        return 0 

def calculate_sliding_window_scores(scores):
    label_score = {idx: [score] for idx, score in enumerate(scores[0].tolist())}
    
    for window in scores:
        for idx, common_class_score in enumerate(window[:args.sliding_window_step].tolist()):
            label_score[(len(label_score)-args.sliding_window_step)+idx].append(common_class_score)
        for class_score in window[args.sliding_window_step:].tolist():
            label_score[len(label_score)] = [class_score]
            
    return torch.tensor([np.mean(score) for score in label_score.values()])
    

def get_real_labels(model_scores, mode):
    eval_labels = os.path.join(args.export_root, f'{mode}_labels.pkl')
    with open(eval_labels, 'rb') as f:
        real_labels = pickle.load(f)
    
    # ranks = []
    # ranked_labels = []
    # i = 0
    # for labels, indices in real_labels:
    #     y = {rid: 0 for rid in labels.keys()}
    #     pairs = list(combinations(labels.keys(), 2))
    #     scores = model_scores[i:i+len(pairs)]
    #     i+=len(pairs)
        
    #     for idx, pair in enumerate(scores):
    #         sorted_pair = torch.zeros_like(pair)
    #         sorted_pair = sorted_pair.index_copy_(0, torch.tensor(indices[idx]).type(torch.int64), pair)
    #         y[pairs[idx][sorted_pair.argmax()]] += 1
            
    #     y = dict(sorted(y.items(), key=lambda item: item[1], reverse=True))
    #     ranks.append(str(y))
    #     y = [labels[rid] for rid in y.keys()]
    #     ranked_labels.append(str(y))
        
    # return ranked_labels, ranks

    ranks = []
    ranked_labels = []
    i=0
    for labels in real_labels:
        scores = model_scores[i:i+len(labels)]
        i+=len(labels)
        
        scores = torch.softmax(scores, dim=0)
        y = {rid: score[0] for rid, score in zip(labels.keys(), scores.tolist())}
        y = dict(sorted(y.items(), key=lambda item: item[1], reverse=True))
        ranks.append(str(y))
        y = [labels[rid] for rid in y.keys()]
        ranked_labels.append(str(y))
        
    return ranked_labels, ranks

def absolute_recall_mrr_ndcg_for_ks(scores, mode, num_candidates, ret_model, summary):
    mode = 'valid' if mode == 'eval' else 'test'
    real_labels, rank = get_real_labels(scores, mode)
    eval_df = pd.DataFrame(real_labels, columns=['y'])
    eval_df['rank'] = rank
    eval_df['y'] = eval_df['y'].apply(eval)
    
    summary = '' if not summary else f'_{summary}'
    y_true_df = pd.read_csv(os.path.join(
        f'/sise/bshapira-group/lilachzi/models/LlamaRec/data/preprocessed/ciao_{args.category.replace(" & ", "_")}_{args.signal}',
        f'{ret_model}_{mode}_{str(num_candidates)}_candidates.csv'),
        converters={'y_true': eval}
    )
    y_true_df = y_true_df[:len(eval_df)]
    eval_df['y_true'] = y_true_df['y_true'].tolist()
    eval_df['y_true'] = eval_df.apply(
        lambda r: list(r['y_true'].values())[len(r['y']):] if len(r['y_true']) > len(r['y']) else [], axis=1
    )
    eval_df['y'] = eval_df['y'] + eval_df['y_true']
    
    eval_df['nDCG@5'] = eval_df.apply(lambda x: calc_ndcg(x['y'][:5]), axis=1)
    eval_df['MRR'] = eval_df.apply(lambda x: calc_mrr(x['y']), axis=1)
    eval_df['Recall@1'] = eval_df.apply(lambda x: calc_recall_at_k(x['y'], x['y'], k=1), axis=1)
    eval_df['Recall@3'] = eval_df.apply(lambda x: calc_recall_at_k(x['y'], x['y'], k=3), axis=1)
    eval_df['Recall@5'] = eval_df.apply(lambda x: calc_recall_at_k(x['y'], x['y'], k=5), axis=1)
    eval_df['Precision@1'] = eval_df.apply(lambda x: calc_precision_at_k(x['y'], k=1), axis=1)
    eval_df['Precision@3'] = eval_df.apply(lambda x: calc_precision_at_k(x['y'], k=3), axis=1)
    eval_df['Precision@5'] = eval_df.apply(lambda x: calc_precision_at_k(x['y'], k=5), axis=1)
    eval_df['Hit@1'] = eval_df.apply(lambda x: calc_hit_at_k(x['y'], k=1), axis=1)
    eval_df['Hit@3'] = eval_df.apply(lambda x: calc_hit_at_k(x['y'], k=3), axis=1)
    eval_df['Hit@5'] = eval_df.apply(lambda x: calc_hit_at_k(x['y'], k=5), axis=1)
    
    if mode == 'test':
        eval_df.to_csv(os.path.join(args.export_root, 'predictions.csv'), index=False)
    else:
        datetime_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
        eval_df.to_csv(os.path.join(args.export_root, f'eval_predictions_{datetime_string}.csv'), index=False)

    metrics = {k:v for k,v in eval_df.iloc[:, 3:].mean().items()}

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
