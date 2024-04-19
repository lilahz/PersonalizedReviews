import math
import os
import wandb
from ast import literal_eval
from tqdm import tqdm

import numpy as np
import pandas as pd

tqdm.pandas()


BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
SHARED_PATH = '/sise/bshapira-group/lilachzi/models'
CSVS_PATH = '/sise/bshapira-group/lilachzi/csvs'
RESULTS_PATH = os.path.join(SHARED_PATH, 'magnn')

def _node_mapping(nodes):
    # id2idx = mapping betweem the node original id and its sorted index
    # idx2id = mapping between the sorted index and the corresponsing original id
    id2idx, idx2id = {}, {}

    for idx, node in enumerate(nodes):
        id2idx[node] = idx
        idx2id[idx] = node

    return id2idx, idx2id

def _get_mappings(category):
    train_set = pd.read_csv(os.path.join(CSVS_PATH, 'train_set.csv'), converters={'y_true': literal_eval})
    train_set = train_set[train_set['category'] == category]
    
    val_set = pd.read_csv(os.path.join(CSVS_PATH, 'valid_set.csv'), converters={'y_true': literal_eval})
    val_set = val_set[val_set['category'] == category]
    
    test_set = pd.read_csv(os.path.join(CSVS_PATH, 'test_set.csv'), converters={'y_true': literal_eval})
    test_set = test_set[test_set['category'] == category]
    
    users = set(train_set['voter_id'].unique()) | set(val_set['voter_id'].unique()) | set(test_set['voter_id'].unique())
    products = set(train_set['product_id'].unique()) | set(val_set['product_id'].unique()) | set(test_set['product_id'].unique())

    reviews = []
    for row in train_set['y_true']:
        reviews.extend(list(row.keys()))
    for row in val_set['y_true']:
        reviews.extend(list(row.keys()))
    for row in test_set['y_true']:
        reviews.extend(list(row.keys()))
    reviews = set(reviews)
    
    users_id2idx, users_idx2id = _node_mapping(users)
    reviews_id2idx, reviews_idx2id = _node_mapping(reviews)
    
    return users_id2idx, users_idx2id, reviews_id2idx, reviews_idx2id

def cosine_sim(u, v):
    """
    Find cosine similarity distance between two vectors
    """
    co_sim = np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v))
    if np.isnan(np.sum(co_sim)):
        return 0 
    return co_sim

def get_y(review_idx, y_true):
    y = []
    
    for idx in review_idx:
        voted_score = y_true[idx]
        y.append(voted_score)
        
    return y

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

def calculate_scores(test_set, task, run_name):
    wandb.init(project='personalized_reviews', name=run_name)

    scores_df = pd.DataFrame()

    scores_df['nDCG@5'] = test_set.apply(lambda row: calc_ndcg(row['y'][:5]), axis=1)
    scores_df['MRR'] = test_set.apply(lambda row: calc_mrr(row['y']), axis=1)
    scores_df['Recall@1'] = test_set.apply(lambda row: calc_recall_at_k(row['y_true'], row['y'], 1), axis=1)
    scores_df['Recall@3'] = test_set.apply(lambda row: calc_recall_at_k(row['y_true'], row['y'], 3), axis=1)
    scores_df['Recall@5'] = test_set.apply(lambda row: calc_recall_at_k(row['y_true'], row['y'], 5), axis=1)
    scores_df['Precision@1'] = test_set.apply(lambda row: calc_precision_at_k(row['y'], 1), axis=1)
    scores_df['Precision@3'] = test_set.apply(lambda row: calc_precision_at_k(row['y'], 3), axis=1)
    scores_df['Precision@5'] = test_set.apply(lambda row: calc_precision_at_k(row['y'], 5), axis=1)
    scores_df['Hit@1'] = test_set.apply(lambda row: calc_hit_at_k(row['y'], 1), axis=1)
    scores_df['Hit@3'] = test_set.apply(lambda row: calc_hit_at_k(row['y'], 3), axis=1)
    scores_df['Hit@5'] = test_set.apply(lambda row: calc_hit_at_k(row['y'], 5), axis=1)

    scores = scores_df.describe()
    print('DEBUG nDCG: ', scores['nDCG@5']['mean'])
    print('DEBUG MRR: ', scores['MRR']['mean'])
    print('DEBUG: Recall@5', scores['Recall@5']['mean'])
    
    wandb.log({
        f'scores/{task}/nDCG': scores['nDCG@5']['mean'],
        f'scores/{task}/Recall@1': scores['Recall@1']['mean'], 
        f'scores/{task}/Recall@3': scores['Recall@3']['mean'], 
        f'scores/{task}/Recall@5': scores['Recall@5']['mean']
    })
    
    results_path = os.path.join(RESULTS_PATH, 'model_results.csv')
    df_results = pd.DataFrame([[run_name] + list(scores_df.mean().values)], 
                              columns=['model'] + list(scores_df.mean().index))
    if os.path.exists(results_path):
        df_results.to_csv(results_path, index=False, mode='a', header=False)
    else:
        df_results.to_csv(results_path, index=False)
    
    return scores_df

def get_link_prediction_recommendation(row, y_proba):
    y_true_dict = row['y_true']
    user_id = row['voter_id']

    scores = {}
    to_remove = []
    for review_id in y_true_dict.keys():
        if not (user_id, review_id) in y_proba:
            to_remove.append(review_id)
            continue
            
        scores[review_id] = y_proba[(user_id, review_id)]
    
    scores = list(dict(sorted(list(scores.items()), key=lambda item: item[1], reverse=True)).keys())

    y_true_dict = {review_id: rating for review_id, rating in y_true_dict.items() if review_id not in to_remove}
    y = get_y(scores, y_true_dict)
    
    return y

def evaluate_link_prediction(pos_user_review, neg_user_review, y_proba, run_name, category):
    test_user_review = np.concatenate([pos_user_review, neg_user_review])
    _, users_idx2id, _, reviews_idx2id = _get_mappings(category)
    
    y_proba_dict = {}
    for i, (user_idx, review_idx) in tqdm(enumerate(test_user_review)):
        user_id = users_idx2id[user_idx]
        review_id = reviews_idx2id[review_idx]
        
        y_proba_dict[(user_id, review_id)] = y_proba[i]
        
    test_set = pd.read_csv(os.path.join(CSVS_PATH, 'test_set.csv'), converters={'y_true': literal_eval})
    test_set = test_set[test_set['category'] == category]
    
    test_set.loc[:, 'y'] = test_set.progress_apply(
        lambda row: get_link_prediction_recommendation(row, y_proba_dict), axis=1
    )
    
    calculate_scores(test_set, 'Link Prediction', run_name)

def get_cosine_similarity_recommendation(row, users_embeddings, review_embeddings):
    y_true_dict = row['y_true']

    user_vector = users_embeddings[row['voter_id']]
    review_embeddings_keys = review_embeddings.keys()
    
    distances = {}
    to_remove = []
    for review_id in y_true_dict.keys():
        if not review_id in review_embeddings_keys:
            to_remove.append(review_id)
            continue
            
        distances[review_id] = cosine_sim(user_vector, review_embeddings[review_id])
        
    distances = list(dict(sorted(list(distances.items()), key=lambda item: item[1], reverse=True)).keys())

    y_true_dict = {review_id: rating for review_id, rating in y_true_dict.items() if review_id not in to_remove}
    y = get_y(distances, y_true_dict)
    
    return y

def evaluate_cosine_similarity(users_embeddings, reviews_embeddings, run_name, category):
    _, users_idx2id, _, reviews_idx2id = _get_mappings(category)
    user_id_embeddings = {}
    for user_idx, embd in tqdm(users_embeddings.items()):
        user_id = users_idx2id[user_idx]
        user_id_embeddings[user_id] = embd

    review_id_embeddings = {}
    for review_idx, embd in tqdm(reviews_embeddings.items()):
        review_id = reviews_idx2id[review_idx]
        review_id_embeddings[review_id] = embd
        
    test_set = pd.read_csv(os.path.join(CSVS_PATH, 'test_set.csv'), converters={'y_true': literal_eval})
    test_set = test_set[test_set['category'] == category]
    
    test_set.loc[:, 'y'] = test_set.progress_apply(
        lambda row: get_cosine_similarity_recommendation(row, user_id_embeddings, review_id_embeddings), axis=1
    )
    
    calculate_scores(test_set, 'Cosine Similarity', run_name)
    
def get_link_classification_recommendation(row, users_id2idx, reviews_id2idx, y_proba):
    y_true_dict = row['y_true']
    scores = {}
    
    user_idx = users_id2idx[row['voter_id']]
    for review_id in y_true_dict:
        review_idx = reviews_id2idx[review_id]
        scores[review_id] = y_proba[(user_idx, review_idx)]

    scores = list(dict(sorted(list(scores.items()), key=lambda item: item[1], reverse=True)).keys())
    
    y = get_y(scores, y_true_dict)
    return y
    
def evaluate_link_classification(y_proba, run_name, category):
    users_id2idx, _, reviews_id2idx, _ = _get_mappings(category)
    
    test_set = pd.read_csv(os.path.join(CSVS_PATH, 'test_set.csv'), converters={'y_true': literal_eval})
    test_set = test_set[test_set['category'] == category]
    
    test_set.loc[:, 'y'] = test_set.progress_apply(
        lambda row: get_link_classification_recommendation(row, users_id2idx, reviews_id2idx, y_proba), axis=1
    )
    
    calculate_scores(test_set, 'Link Classification', run_name)
