import logging.handlers
import math
import numpy as np
import os
import re
from string import punctuation

import torch
import nltk
from nltk.corpus import stopwords


stopwords_list = stopwords.words("english")


def get_logger(log_file_name):
    # Set logger configurations
    fh = logging.handlers.RotatingFileHandler(filename=log_file_name, maxBytes=10485760, backupCount=10)
    # Add log formatter with timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # create a file handler
    logger = logging.getLogger(log_file_name)
    logger.addHandler(fh)
    # add the file handler to the logger
    logger.setLevel(logging.DEBUG)
    return logger


def verify_folder(file_path):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    except Exception as e:
        print("Exception in utils.verify_folders: " + str(e))


def is_ordinal(ch):
    if ord(ch) <= 128:
        return ch
    return " "


def find_chinese_character(sentence):
    if re.search(u'[\u4e00-\u9fff]', sentence):
        return 1
    return 0


def clean_text(title, lower=True, punct=True):
    if lower:
        title = title.lower()
    title = "".join([is_ordinal(ch) for ch in title])
    if punct:
        title = "".join([ch for ch in title if ch not in punctuation])
    title = " ".join(title.split()).strip()
    return title


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


def clean_review(review):
    review = review.lower()
    # Remove non Ascii letters
    review = "".join(c for c in review if ord(c) < 128)
    # Clean text from special characters
    review = re.sub('[^A-Za-z0-9 ]+', ' ', review.strip())
    # Remove stopwords
    # review = " ".join([w for w in review.split() if w not in stopwords_list])
    # Remove spaces
    # review = " ".join(review.split())

    return review


def save_checkpoint(save_path, model_name, model):
    model_path = os.path.join(save_path, model_name + '.pt')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to ==> {model_path}')


def load_checkpoint(load_path, model_name, device, model):
    model_path = os.path.join(load_path, model_name + '.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f'Model loaded from <== {model_path}')
    return model

def eval_array(array_str):
    # first remove space after first square bracket
    array_str = array_str.replace('[ ', '[')
    # then replace spaces and new lines with comma
    array_str = re.sub(r'\s+', ',', array_str)
    # convert the string to numpy array
    array = np.array(eval(array_str))
    
    return array