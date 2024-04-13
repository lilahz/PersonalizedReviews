from config import *

import json
import math
import os
import pandas as pd
import pickle
import pprint as pp
import random
from datetime import datetime
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
    mode = 'valid' if mode == 'eval' else 'test'
    eval_df_path = os.path.join(CSVS_FOLDER, f'{mode}_set.csv')
    eval_df = pd.read_csv(eval_df_path, converters={'y_true': eval})
    eval_df = eval_df[eval_df['category'] == args.category]
    eval_df['votes'] = eval_df['y_true'].apply(lambda d: [(k,v) for k,v in d.items()])
    eval_df = eval_df.explode('votes')
    eval_df['review_id'] = eval_df['votes'].apply(lambda l: l[0])
    eval_df['vote'] = eval_df['votes'].apply(lambda l: l[1])
    eval_df.drop(columns=['y_true', 'votes'], inplace=True)
    
    eval_df = eval_df[:len(model_scores)]
    eval_df['prediction'] = model_scores[:, 0].tolist()
    
    eval_df = eval_df.sort_values(['product_id', 'voter_id', 'prediction'], ascending=[0, 0, 0]).groupby(
        ['product_id', 'voter_id'], as_index=False).agg({'review_id': list, 'vote': list, 'prediction': list}).rename(
            columns={'vote': 'y'})
    
    return eval_df

def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks, mode):
    eval_df = get_real_labels(scores, mode)
    
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

    metrics = {k:v for k,v in eval_df.iloc[:, 5:].mean().items()}

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
