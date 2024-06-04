from .base import AbstractDataset
from .utils import *

from datetime import date
from pathlib import Path
import pickle
import shutil
import tempfile
import os
import random

import gzip
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


class CiaoDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'ciao'
    
    @classmethod
    def url(cls):
        pass

    @classmethod
    def zip_file_content_is_folder(cls):
        pass
    
    @classmethod
    def all_raw_file_names(cls):
        return ['votes_4core_split_60_20_20.tsv', '../csvs/ciao_4core_gpt_aspects.csv']
    
    def maybe_download_raw_dataset(self):
        folder_path = Path('/sise/home/lilachzi/PersonzalizedReviews/csvs')
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        
    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        print(f'DEBUG: dataset_path: {dataset_path}')
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        self.maybe_download_raw_dataset()
        df, umap, smap = self.load_ratings_df()
        meta_raw = self.load_meta_dict()
        like_history, write_history, umap, smap = self.calculcate_user_history(umap, smap)
        train, val, test = self.split_df(df, like_history, write_history)
        meta = {smap[k]: v for k, v in meta_raw.items() if k in smap}
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'meta': meta,
                   'umap': umap,
                   'smap': smap}
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)
        
    def load_ratings_df(self):
        folder_path = Path('/sise/home/lilachzi/PersonzalizedReviews/csvs')
        file_path = folder_path.joinpath(self.all_raw_file_names()[0])
        df = pd.read_csv(file_path, sep='\t')
        
        # create user / reviews mappings
        uid2idx = {u: i for i, u in enumerate(set(set(df['voter_id']) | set(df['user_id'])), start=1)}
        sid2idx = {s: i for i, s in enumerate(set(df['review_id']), start=1)}
        
        df = df[df['category'] == self.category][['voter_id', 'review_id', 'vote', 'split']]
        df.rename(columns={'voter_id': 'uid', 'review_id': 'sid', 'vote': 'rating'}, inplace=True)
        
        df['uid'] = df['uid'].map(uid2idx)
        df['sid'] = df['sid'].map(sid2idx)
        
        return df, uid2idx, sid2idx
    
    def load_meta_dict(self):
        folder_path = Path('/sise/home/lilachzi/PersonzalizedReviews/csvs')
        file_path = folder_path.joinpath(self.all_raw_file_names()[1])
        
        df = pd.read_csv(file_path)
        df = df[df['category'] == self.category]
        meta_dict = dict(zip(df.review_id, df.aspects))
        
        return meta_dict
    
    def calculcate_user_history(self, umap, smap):
        folder_path = Path('/sise/home/lilachzi/PersonzalizedReviews/csvs')
        positive_vote = [3, 4, 5]
        
        written_review = {}
        liked_reviews = {}
        
        votes_df = pd.read_csv(folder_path.joinpath(self.all_raw_file_names()[0]), sep='\t')
        for idx, row in tqdm(votes_df.iterrows(), total=len(votes_df)):
            review_id = smap[row['review_id']]
            voter_id = umap[row['voter_id']]
            vote = row['vote']
            split = row['split']
            categ = row['category']
            
            if vote in positive_vote:
                if voter_id not in liked_reviews:
                    liked_reviews[voter_id] = {}
                if split not in liked_reviews[voter_id]:
                    liked_reviews[voter_id][split] = []
                liked_reviews[voter_id][split].append((review_id, categ))
                
        reviews_df = pd.read_csv(folder_path.joinpath(self.all_raw_file_names()[1]))
        for idx, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
            if row['review_id'] not in smap:
                smap[row['review_id']] = len(smap)
            review_id = smap[row['review_id']]
            if row['user_id'] not in umap:
                umap[row['user_id']] = len(umap)
            user_id = umap[row['user_id']]
            if user_id not in written_review:
                written_review[user_id] = []
            written_review[user_id].append(review_id)

        return liked_reviews, written_review, umap, smap
    
    def split_df(self, df, like_history, write_history):
        print('Splitting')
        user2items = df[(df['rating'] >= 3) & (df['split'] == 'train')].groupby('uid', as_index=False).agg({'sid': list})
            
        train, val, test = {}, {}, {}
        for user in user2items['uid'].unique():
            try:
                items = random.sample(user2items[user2items['uid'] == user]['sid'].item(), 2)
            except:
                continue
            
            history = self.load_history(user, like_history, write_history, 'train')
            train[user], val[user], test[user] = history, [items[0]], [items[1]]
        return train, val, test
    
    def load_history(self, voter_id, liked_reviews, written_review, split):
        history_write = written_review[voter_id]
        if split in liked_reviews.get(voter_id, []):
            liked_reviews_user = liked_reviews[voter_id][split]
        else:
            liked_reviews_user = []
            
        history_like = [liked_pair[0] for liked_pair in liked_reviews_user if liked_pair[1] != self.category]

        if self.signal == 'write':
            history = np.random.permutation(history_write)[:15]
        elif self.signal == 'like':
            history = np.random.permutation(history_like)[:30]
        else:  # 'write-like'
            history = (list(np.random.permutation(history_write)) +
                    list(np.random.permutation(history_like)))

        return history
