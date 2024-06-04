import os
import random
import re
import sys

current_dir = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import torch
from gensim.models.keyedvectors import load_word2vec_format
from torch.utils.data import Dataset
from tqdm import tqdm

from config import hparams
from commons import Commons

# todo: use embedding of other models like roberta, universal sentence
# todo: try with and without stopwords
clean_reviews_file = os.path.join(Commons.CSVS_PATH, 'ciao_4core.tsv')
llama_reviews_file = os.path.join(Commons.CSVS_PATH, 'ciao_4core_summary.tsv')
davinci_reviews_file = os.path.join(Commons.CSVS_PATH, 'ciao_4core_summary_davinci.tsv')
gpt_aspects_reviews_file = os.path.join(Commons.CSVS_PATH, 'ciao_4core_gpt_aspects.tsv')
votes_file = os.path.join(Commons.CSVS_PATH, 'votes_4core_split_60_20_20.tsv')
if hparams['use_candidates']:
    candidates_path = '/sise/bshapira-group/lilachzi/models/LlamaRec/data/preprocessed/'
    candidates_path = os.path.join(candidates_path, f"ciao_{hparams['candidates_category']}_{hparams['his_type']}")
    train_votes_file = os.path.join(Commons.CSVS_PATH, 'train_set.csv')
    valid_votes_file = os.path.join(candidates_path, f'magnn_valid_{str(hparams["num_candidates"])}_candidates.csv')
    test_votes_file = os.path.join(candidates_path, f'magnn_test_{str(hparams["num_candidates"])}_candidates.csv')
else:
    train_votes_file = os.path.join(Commons.CSVS_PATH, 'train_set.csv')
    valid_votes_file = os.path.join(Commons.CSVS_PATH, 'valid_set.csv')
    test_votes_file = os.path.join(Commons.CSVS_PATH, 'test_set.csv')
train_method_dict = {'1': 'product_reviews_votes', '2': 'product_reviews_date', '3': 'all_reviews_votes', '4': 'eval'}


def word_tokenize(text):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?]")
    if isinstance(text, str):
        return pat.findall(text.lower())
    else:
        return []


def newsample(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def load_data(embedding_model, llm_summary, col_spliter="\t"):
    """
    init reviews information given reviews file.
    Args:
        split: which split of the data to read (train / valid / test)
    """

    positive_vote = [3, 4, 5]
    review_size = hparams['review_size']
    reviews = [""]
    word_dict = {}
    rid2index = {0: 0}
    written_review = {}
    liked_reviews = {}

    with open(votes_file, "r") as rd:
        for line in rd:
            _, review_id, _, voter_id, vote, _, categ, split = line.strip("\n").split(col_spliter)
            if review_id == 'review_id':
                continue
            review_id = int(review_id)
            voter_id = int(voter_id)
            vote = int(vote)
            if vote in positive_vote:
                if voter_id not in liked_reviews:
                    liked_reviews[voter_id] = {}
                if split not in liked_reviews[voter_id]:
                    liked_reviews[voter_id][split] = []
                liked_reviews[voter_id][split].append((review_id, categ))

    if llm_summary == 'llama':
        reviews_file = llama_reviews_file
    elif llm_summary == 'davinci':
        reviews_file = davinci_reviews_file
    elif llm_summary == 'gpt_aspects':
        reviews_file = gpt_aspects_reviews_file
    else:
        reviews_file = clean_reviews_file
    with open(reviews_file, "r") as rd:
        for line in rd:
            if not llm_summary:
                _, review_id, user_id, _, _, _, _, text, _ = line.strip("\n").split(col_spliter)
            else:
                if llm_summary == 'gpt_aspects':
                    _, review_id, user_id, _, _, _, _, _, _, text = line.strip("\n").split(col_spliter)
                else:
                    _, review_id, user_id, _, _, _, text, summary, _, text_len = line.strip("\n").split(col_spliter)
            if review_id == 'review_id':
                continue
            review_id = int(review_id)
            user_id = int(user_id)
            if user_id not in written_review:
                written_review[user_id] = []
            written_review[user_id].append(review_id)

            if review_id in rid2index:
                continue

            rid2index[review_id] = len(rid2index)
            if llm_summary and int(text_len) >= 500:
                text = summary
            review = word_tokenize(text)
            reviews.append(review)

    num_reviews = len(reviews)

    if hparams['embedding_type'] == 'llm':
        review_text_index = np.zeros((num_reviews, review_size), dtype='int32')
        review_text_attn = np.zeros((num_reviews, review_size), dtype='int32')
        for review_index in range(len(reviews)):
            review = reviews[review_index]
            review_res = embedding_model(" ".join(review), padding='max_length', truncation=True)
            review_text_index[review_index] = review_res['input_ids']
            review_text_attn[review_index] = review_res['attention_mask']
            
        return rid2index, written_review, liked_reviews, review_text_index, review_text_attn
    
    else:
        review_text_index = np.zeros((num_reviews, review_size), dtype="float64")
        unknown = {}
        for review_index in range(len(reviews)):
            review = reviews[review_index]
            for word_index in range(min(review_size, len(review))):
                if review[word_index] in embedding_model:
                    if review[word_index] not in word_dict:
                        word_dict[review[word_index]] = embedding_model.get_index(review[word_index])
                    review_text_index[review_index, word_index] = word_dict[review[word_index].lower()]
                else:
                    if review[word_index] in unknown:
                        unknown[review[word_index]] += 1
                    else:
                        unknown[review[word_index]] = 1

        return rid2index, written_review, liked_reviews, review_text_index


def load_history(voter_id, rid2index, written_review, liked_reviews, split, history_type, category=None):
    history_write = [rid2index[i] for i in written_review[voter_id]]
    if split in liked_reviews.get(voter_id, []):
        liked_reviews_user = liked_reviews[voter_id][split]
    else:
        liked_reviews_user = []
    if category:
        history_like = [rid2index[liked_pair[0]] for liked_pair in liked_reviews_user if liked_pair[1] != category]
    else:
        history_like = [rid2index[liked_pair[0]] for liked_pair in liked_reviews_user]

    if history_type == 'write':
        history = np.random.permutation(history_write)[:hparams['his_size']]
    elif history_type == 'like':
        history = np.random.permutation(history_like)[:hparams['his_size']]
    else:  # 'combined'
        history = (list(np.random.permutation(history_write)[:hparams['write_his_size']]) +
                   list(np.random.permutation(history_like)[:hparams['like_his_size']]))

    history = list(history) + [0] * (hparams['his_size'] - len(history))
    return history


def load_split_data(rid2index, written_reviews, liked_reviews, split, train_method, category=None, col_spliter="\t"):
    """
    init examples information given votes file.
    Args:
        split: which split of the data to read (train / valid / test)
    """
    if split == 'train':
        votes_file = train_votes_file
    elif split == 'valid':
        votes_file = valid_votes_file
    else:  # 'test'
        votes_file = test_votes_file

    positive_vote = [3, 4, 5]
    histories = []
    impressions = []
    labels = []

    votes_df = pd.read_csv(votes_file)
    votes_df['y_true'] = votes_df['y_true'].apply(eval)
    
    if category and 'category' in votes_df:
        votes_df = votes_df[votes_df['category'] == category]
    else:
        votes_df['candidates'] = votes_df['candidates'].apply(eval)

    # if train_method_dict[train_method] == 'all_reviews_votes':
    user_votes = votes_df.groupby('voter_id').apply(lambda gb: gb['y_true'].tolist()).reset_index().rename(
        columns={0: 'list_dicts'})
    user_votes['like_votes'] = user_votes['list_dicts'].apply(
        lambda li: [k for d in li for k, v in d.items() if v >= 3])
    user_votes = user_votes.set_index('voter_id')[['like_votes']]
    # if train_method_dict[train_method] == 'product_reviews_date':
    # reviews_df = pd.read_csv(reviews_file, sep=col_spliter)
    # reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    # product_reviews_by_date = reviews_df.groupby(['product_id']).apply(
    #     lambda gb_df: gb_df.sort_values('date', ascending=True))[['review_id', 'date']].groupby(['product_id']).agg(
    #     {'review_id': list})
    product_reviews_by_date = None

    count = 0
    for line in votes_df.itertuples():
        if split != 'train' and hparams['use_candidates']:
            product_id, voter_id, review_labels = line[1], line[2], line[3]
        else:
            product_id, voter_id, review_labels = line[2], line[3], line[4]
        history = load_history(voter_id, rid2index, written_reviews, liked_reviews, split, hparams['his_type'],
                               category)

        poss = []
        for review, vote in review_labels.items():
            if vote in positive_vote:
                poss.append(review)

        if train_method_dict[train_method] == 'product_reviews_votes':
            # Select negatives from the set of product reviews that the user didn't like.
            negs = []
            for review, vote in review_labels.items():
                if review not in poss:
                    negs.append(review)

            for p in poss:
                n = newsample(negs, hparams['npratio'])
                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1] + [0] * hparams['npratio']
                histories.append(history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        elif train_method_dict[train_method] == 'product_reviews_date':
            # Select negatives from the set of product reviews that the user didn't like and were presented together
            # with the positive sample in the product page (their date is earlier than the date of the positive).
            reviews_by_date = product_reviews_by_date.loc[product_id]['review_id']
            for p in poss:
                poss_idx = reviews_by_date.index(p)
                negs = []
                for review_id in reviews_by_date[:poss_idx]:
                    if review_id not in poss:
                        negs.append(review_id)
                if len(negs) == 0:
                    continue
                n = newsample(negs, 1)
                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1, 0]
                histories.append(history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        elif train_method_dict[train_method] == 'all_reviews_votes':
            # Select negatives from all products reviews that the user didn't like.
            liked_reviews = user_votes.loc[voter_id]
            negs = list(rid2index.keys())
            for p in poss:
                n = []
                while len(n) < hparams['npratio']:
                    sample_neg = newsample(negs, 1)[0]
                    if sample_neg not in liked_reviews:
                        n.append(sample_neg)

                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1] + [0] * hparams['npratio']
                histories.append(history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        else:
            product_reviews = [rid2index[k] for k, v in review_labels.items()]
            votes_label = [v for k, v in review_labels.items()]
            histories.append(history)
            impressions.append(product_reviews)
            labels.append(votes_label)

        count += 1
        # if count > 100:
        #     break

    return histories, impressions, labels


def load_split_data_combined(rid2index, written_reviews, liked_reviews, split, train_method, category=None,
                             col_spliter="\t"):
    """
    init examples information given votes file.
    Args:
        split: which split of the data to read (train / valid / test)
    """
    if split == 'train':
        votes_file = train_votes_file
    elif split == 'valid':
        votes_file = valid_votes_file
    else:  # 'test'
        votes_file = test_votes_file

    positive_vote = [3, 4, 5]
    write_histories = []
    like_histories = []
    impressions = []
    labels = []

    votes_df = pd.read_csv(votes_file)
    votes_df['y_true'] = votes_df['y_true'].apply(eval)
    
    if category and 'category' in votes_df:
        votes_df = votes_df[votes_df['category'] == category]
    else:
        votes_df['candidates'] = votes_df['candidates'].apply(eval)

    # if train_method_dict[train_method] == 'all_reviews_votes':
    user_votes = votes_df.groupby('voter_id').apply(lambda gb: gb['y_true'].tolist()).reset_index().rename(
        columns={0: 'list_dicts'})
    user_votes['like_votes'] = user_votes['list_dicts'].apply(
        lambda li: [k for d in li for k, v in d.items() if v >= 3])
    user_votes = user_votes.set_index('voter_id')[['like_votes']]
    # if train_method_dict[train_method] == 'product_reviews_date':
    # reviews_df = pd.read_csv(reviews_file, sep=col_spliter)
    # reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    # product_reviews_by_date = reviews_df.groupby(['product_id']).apply(
    #     lambda gb_df: gb_df.sort_values('date', ascending=True))[['review_id', 'date']].groupby(['product_id']).agg(
    #     {'review_id': list})
    product_reviews_by_date = None

    count = 0
    for line in votes_df.itertuples():
        if split != 'train' and hparams['use_candidates']:
            product_id, voter_id, review_labels = line[1], line[2], line[3]
        else:
            product_id, voter_id, review_labels = line[2], line[3], line[4]

        hparams['his_size'] = hparams['write_his_size']
        write_history = load_history(voter_id, rid2index, written_reviews, liked_reviews, split, 'write', category)
        hparams['his_size'] = hparams['like_his_size']
        like_history = load_history(voter_id, rid2index, written_reviews, liked_reviews, split, 'like', category)

        poss = []
        for review, vote in review_labels.items():
            if vote in positive_vote:
                poss.append(review)

        if train_method_dict[train_method] == 'product_reviews_votes':
            # Select negatives from the set of product reviews that the user didn't like.
            negs = []
            for review, vote in review_labels.items():
                if review not in poss:
                    negs.append(review)

            for p in poss:
                n = newsample(negs, hparams['npratio'])
                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1] + [0] * hparams['npratio']
                write_histories.append(write_history)
                like_histories.append(like_history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        elif train_method_dict[train_method] == 'product_reviews_date':
            # Select negatives from the set of product reviews that the user didn't like and were presented together
            # with the positive sample in the product page (their date is earlier than the date of the positive).
            reviews_by_date = product_reviews_by_date.loc[product_id]['review_id']
            for p in poss:
                poss_idx = reviews_by_date.index(p)
                negs = []
                for review_id in reviews_by_date[:poss_idx]:
                    if review_id not in poss:
                        negs.append(review_id)
                if len(negs) == 0:
                    continue
                n = newsample(negs, 1)
                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1, 0]
                write_histories.append(write_history)
                like_histories.append(like_history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        elif train_method_dict[train_method] == 'all_reviews_votes':
            # Select negatives from all products reviews that the user didn't like.
            liked_reviews = user_votes.loc[voter_id]
            negs = list(rid2index.keys())
            for p in poss:
                n = []
                while len(n) < hparams['npratio']:
                    sample_neg = newsample(negs, 1)[0]
                    if sample_neg not in liked_reviews:
                        n.append(sample_neg)

                impr = [p] + n
                product_reviews = [rid2index[i] for i in impr]
                votes_label = [1] + [0] * hparams['npratio']
                write_histories.append(write_history)
                like_histories.append(like_history)
                impressions.append(product_reviews)
                labels.append(votes_label)

        else:
            product_reviews = [rid2index[k] for k, v in review_labels.items()]
            votes_label = [v for k, v in review_labels.items()]
            write_histories.append(write_history)
            like_histories.append(like_history)
            impressions.append(product_reviews)
            labels.append(votes_label)

        count += 1
        # if count > 100:
        #     break

    return write_histories, like_histories, impressions, labels


class ReviewsDataset(Dataset):
    def __init__(self, embedding_model, llm_summary, embedding_type='w2v'):
        if embedding_type == 'llm':
            self.rid2index, self.written_reviews, self.liked_reviews, self.review_text_index, self.review_text_attn = (
                load_data(embedding_model, llm_summary))
        else:
            self.rid2index, self.written_reviews, self.liked_reviews, self.review_text_index = load_data(embedding_model, llm_summary)


class TrainDataset(Dataset):
    def __init__(self, reviews_dataset, combined=False, category=None):
        """Initialize the dataset.
        Args:
            hparams (object): Global hyper-parameters. Some key settings such as head_num and head_dim are there.
            npratio (int): negative and positive ratio used in negative sampling. -1 means no need of negative sampling.
            col_spliter (str): column spliter in one line.
        """
        self.train_method = hparams['train_method']
        self.combined = combined
        self.llm_embedding = hparams['embedding_type'] == 'llm'
        
        if self.llm_embedding:
            (rid2index, written_reviews, liked_reviews, self.review_text_index,
            self.review_text_attn) = (reviews_dataset.rid2index,
                                        reviews_dataset.written_reviews,
                                        reviews_dataset.liked_reviews,
                                        reviews_dataset.review_text_index,
                                        reviews_dataset.review_text_attn)
        else:
            rid2index, written_reviews, liked_reviews, self.review_text_index = (reviews_dataset.rid2index,
                                                                             reviews_dataset.written_reviews,
                                                                             reviews_dataset.liked_reviews,
                                                                             reviews_dataset.review_text_index)

        if self.combined:
            self.write_histories, self.like_histories, self.imprs, self.labels = load_split_data_combined(
                rid2index,
                written_reviews,
                liked_reviews,
                split='train',
                train_method=self.train_method,
                category=category)
        else:
            self.histories, self.imprs, self.labels = load_split_data(rid2index,
                                                                      written_reviews,
                                                                      liked_reviews,
                                                                      split='train',
                                                                      train_method=self.train_method,
                                                                      category=category)

    def __len__(self):
        return len(self.imprs)

    def __getitem__(self, idx):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negative sampled result.

        Args:
            idx (int): sample index.

        Return:
            history  (tensor): [batch, num_click_docs, seq_len]
            candidates (tensor): [batch, num_candidate_docs, seq_len]
            candidates_label: labels of candidates
        """

        impr_indices = self.imprs[idx] # history
        
        candidates_label = self.labels[idx]
        if self.combined:
            write_history = self.review_text_index[self.write_histories[idx]]
            like_history = self.review_text_index[self.like_histories[idx]]
        else:
            history = self.review_text_index[self.histories[idx]]

        # Shuffle order
        tmp = list(zip(impr_indices, candidates_label))
        random.shuffle(tmp)
        impr_indices, candidates_label = map(list, zip(*tmp))
        candidates = self.review_text_index[impr_indices]

        # Convert to tensors
        if self.combined:
            write_history = np.asarray(write_history, dtype=np.int64)
            like_history = np.asarray(like_history, dtype=np.int64)
        else:
            history = np.asarray(history, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates_label = np.asarray(candidates_label, dtype=np.float32)
        
        if self.llm_embedding:
            candidates_attn = np.asarray(self.review_text_attn[impr_indices], dtype=np.int32)
            history_attn = np.asarray(self.review_text_attn[self.histories[idx]], dtype=np.int32)
            
            return torch.tensor(history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label), \
                torch.tensor(history_attn), \
                torch.tensor(candidates_attn)

        if self.combined:
            return torch.tensor(write_history), \
                torch.tensor(like_history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label)
        else:
            return torch.tensor(history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label)


class EvalDataset(Dataset):
    def __init__(self, reviews_dataset, split, combined=False, category=None):
        self.train_method = '4'
        self.combined = combined
        self.llm_embedding = hparams['embedding_type'] == 'llm'
        
        if self.llm_embedding:
            (rid2index, written_reviews, liked_reviews, self.review_text_index,
            self.review_text_attn) = (reviews_dataset.rid2index,
                                        reviews_dataset.written_reviews,
                                        reviews_dataset.liked_reviews,
                                        reviews_dataset.review_text_index,
                                        reviews_dataset.review_text_attn)
        else:
            rid2index, written_reviews, liked_reviews, self.review_text_index = (reviews_dataset.rid2index,
                                                                             reviews_dataset.written_reviews,
                                                                             reviews_dataset.liked_reviews,
                                                                             reviews_dataset.review_text_index)

        if self.combined:
            self.write_histories, self.like_histories, self.imprs, self.labels = load_split_data_combined(
                rid2index,
                written_reviews,
                liked_reviews,
                split=split,
                train_method=self.train_method,
                category=category)
        else:
            self.histories, self.imprs, self.labels = load_split_data(rid2index,
                                                                      written_reviews,
                                                                      liked_reviews,
                                                                      split=split,
                                                                      train_method=self.train_method,
                                                                      category=category)

    def __len__(self):
        return len(self.imprs)

    def __getitem__(self, idx):
        """Parse one behavior sample into feature values.

        Args:
            idx (int): sample index.

        Return:
            history  (tensor): [batch, num_click_docs, seq_len]
            candidates (tensor): [batch, num_candidate_docs, seq_len]
            candidates_label: labels of candidates
        """

        impr_indices = self.imprs[idx]
        candidates_label = self.labels[idx]
        candidates = self.review_text_index[impr_indices]
        if self.combined:
            write_history = self.review_text_index[self.write_histories[idx]]
            like_history = self.review_text_index[self.like_histories[idx]]
        else:
            history = self.review_text_index[self.histories[idx]]

        # Convert to tensors
        if self.combined:
            write_history = np.asarray(write_history, dtype=np.int64)
            like_history = np.asarray(like_history, dtype=np.int64)
        else:
            history = np.asarray(history, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates_label = np.asarray(candidates_label, dtype=np.float32)
        
        if self.llm_embedding:
            candidates_attn = np.asarray(self.review_text_attn[impr_indices], dtype=np.int32)
            history_attn = np.asarray(self.review_text_attn[self.histories[idx]], dtype=np.int32)
            
            return torch.tensor(history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label), \
                torch.tensor(history_attn), \
                torch.tensor(candidates_attn)

        if self.combined:
            return torch.tensor(write_history), \
                torch.tensor(like_history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label)
        else:
            return torch.tensor(history), \
                torch.tensor(candidates), \
                torch.tensor(candidates_label)


if __name__ == '__main__':
    w2v = load_word2vec_format(hparams['pretrained_model'], no_header=True)
    train_ds = TrainDataset(w2v)
    print(train_ds[10])

    valid_ds = EvalDataset(w2v, split='valid')
    print(valid_ds[10])
