import time
import argparse
import os
import pickle
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.data import load_ciao_data, get_metapaths_info, load_mappings
from utils.tools import index_generator, parse_minibatch_Ciao
from model import MAGNN_lp

tqdm.pandas()

BASE_PATH = '/sise/bshapira-group/lilachzi/models'
CSVS_PATH = '/sise/bshapira-group/lilachzi/csvs'
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'magnn/checkpoint')
RESULTS_PATH = os.path.join(BASE_PATH, 'magnn/data/results')
OUTPUT_PATH = os.path.join(BASE_PATH, 'LlamaRec/data/preprocessed')

# Params
num_ntype = 3
# Beauty
# num_user = 4417
# num_review = 14091
# Games
# num_user = 4269
# num_review = 8919
# Food & Drink
num_user = 4410
num_review = 12084
# DVDs
# num_user = 5189
# num_review = 27964
# Internet
# num_user = 4965
# num_review = 11737

num_candidates = 10


def build_predictions_df(category, signal, mode, y_proba, user_review, umap, rmap):
    y_proba_dict = {}
    for i, (user_idx, review_idx) in tqdm(enumerate(user_review), total=len(user_review)):
        user_id = umap[user_idx]
        review_id = rmap[review_idx]
        
        y_proba_dict[(user_id, review_id)] = y_proba[i]
        
    df = pd.read_csv(os.path.join(CSVS_PATH, f'{mode}_set.csv'), converters={'y_true': eval})
    df = df[df['category'] == category]
    df['votes'] = df['y_true'].progress_apply(lambda d: [(key, val) for key, val in d.items()])
    df = df.explode('votes')
    df['review_id'] = df['votes'].progress_apply(lambda x: x[0])
    df['vote'] = df['votes'].progress_apply(lambda x: x[1])
    df.drop(columns=['votes'], inplace=True)
    df['prediction'] = df.progress_apply(lambda r: y_proba_dict[(r['voter_id'], r['review_id'])], axis=1)
    candidates = df.sort_values(['product_id', 'voter_id', 'prediction'], ascending=[0, 0, 0]).groupby(
        ['product_id', 'voter_id'], as_index=False).agg({'review_id': lambda x: list(x), 'vote': lambda x: list(x)})
    candidates['candidates'] = candidates.progress_apply(
        lambda row: str(dict(zip(row['review_id'][:num_candidates], row['vote'][:num_candidates]))), axis=1
    )
    candidates.drop(columns={'review_id', 'vote'}, inplace=True)
    candidates['y_true'] = candidates.progress_apply(
        lambda row: df[(df['product_id'] == row['product_id']) & (df['voter_id'] == row['voter_id'])]['y_true'].iloc[0], axis=1
    )
    candidates.to_csv(os.path.join(OUTPUT_PATH, f'ciao_{category.replace(" & ", "_")}_{signal}', f'magnn_{mode}_{str(num_candidates)}_candidates.csv'), index=False)

def run_model_Ciao(category, signal, hidden_dim, num_heads, attn_vec_dim, rnn_type, batch_size, neighbor_samples, **kwargs):
    
    model_results = os.path.join(BASE_PATH, 'magnn', 'model_info_results.csv')
    model_results = pd.read_csv(model_results).fillna('')
    model_results = model_results[(model_results['category'] == category) & (model_results['signal'] == signal)]
    best_model = model_results.iloc[model_results['nDCG@5'].argmax()]
    
    dropout_rate = best_model['dropout_rate']
    lr = best_model['lr']
    weight_decay = best_model['weight_decay']
    feats_type = best_model['feats_type']
    sub_feats_type = f"{best_model['sub_feats_type']}"
    model_name = best_model['model']
    print(f'Best model: {model_name}')
    
    adjlists_ur, edge_metapath_indices_list_ur, _, type_mask, train_val_test_pos_user_review, \
        train_val_test_neg_user_review, features = load_ciao_data(category, signal, feats_type, sub_feats_type)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    else:
        features_list = [torch.FloatTensor(features_type).to(device) for features_type in features.values()]
        in_dims = [features.shape[1] for features in features_list]
        
    val_pos_user_review = train_val_test_pos_user_review['val_pos_user_review']
    val_neg_user_review = train_val_test_neg_user_review['val_neg_user_review']
    
    test_pos_user_review = train_val_test_pos_user_review['test_pos_user_review']
    test_neg_user_review = train_val_test_neg_user_review['test_neg_user_review']
    
    umap, rmap, pmap = load_mappings(category, signal)
    etypes_lists, num_metapaths_list, num_edge_type, use_masks, no_masks = get_metapaths_info(signal)
    
    net = MAGNN_lp(
        num_metapaths_list, num_edge_type, etypes_lists, in_dims, hidden_dim, hidden_dim, 
        num_heads, attn_vec_dim, rnn_type, dropout_rate
    )
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, f'checkpoint_{model_name}.pt')))
    
    pos_val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_review), shuffle=False)
    neg_val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_neg_user_review), shuffle=False)
    
    pos_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_review), shuffle=False)
    neg_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_neg_user_review), shuffle=False)
            
    val_pos_proba_list = []
    val_neg_proba_list = []
    test_pos_proba_list = []
    test_neg_proba_list = []
    net.eval()
    with torch.no_grad():
        # validation
        for iteration in tqdm(range(pos_val_idx_generator.num_iterations()), desc='Positive validation candidates'):
            # forward
            val_idx_batch = pos_val_idx_generator.next()
            val_pos_user_review_batch = val_pos_user_review[val_idx_batch].tolist()
            val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                adjlists_ur, edge_metapath_indices_list_ur, val_pos_user_review_batch, device, neighbor_samples, no_masks, num_user)

            [pos_embedding_user, pos_embedding_review], _ = net(
                (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
            
            pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
            pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)

            pos_out = torch.bmm(pos_embedding_user, pos_embedding_review).flatten()
            val_pos_proba_list.append(torch.sigmoid(pos_out))
            
        for iteration in tqdm(range(neg_val_idx_generator.num_iterations()), desc='Negative validation candidates'):
            # forward
            val_idx_batch = neg_val_idx_generator.next()
            val_neg_user_review_batch = val_neg_user_review[val_idx_batch].tolist()
            
            val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                adjlists_ur, edge_metapath_indices_list_ur, val_neg_user_review_batch, device, neighbor_samples, no_masks, num_user)
            
            [neg_embedding_user, neg_embedding_review], _ = net(
                (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                
            neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
            neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
            
            neg_out = torch.bmm(neg_embedding_user, neg_embedding_review).flatten()
            val_neg_proba_list.append(torch.sigmoid(neg_out))
                    
        y_proba_val = torch.cat(val_pos_proba_list + val_neg_proba_list)
        y_proba_val = y_proba_val.cpu().numpy()
        val_user_review = np.concatenate([val_pos_user_review, val_neg_user_review])
        
        build_predictions_df(category, signal, 'valid', y_proba_val, val_user_review, umap, rmap)

        for iteration in tqdm(range(pos_test_idx_generator.num_iterations()), desc='Positive test candidates'):
            # forward
            test_idx_batch = pos_test_idx_generator.next()
            test_pos_user_review_batch = test_pos_user_review[test_idx_batch].tolist()
            test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                adjlists_ur, edge_metapath_indices_list_ur, test_pos_user_review_batch, device, neighbor_samples, no_masks, num_user)

            [pos_embedding_user, pos_embedding_review], _ = net(
                (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
            
            pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
            pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)

            pos_out = torch.bmm(pos_embedding_user, pos_embedding_review).flatten()
            test_pos_proba_list.append(torch.sigmoid(pos_out))
            
        for iteration in tqdm(range(neg_test_idx_generator.num_iterations()), desc='Negative test candidates'):
            # forward
            test_idx_batch = neg_test_idx_generator.next()
            test_neg_user_review_batch = test_neg_user_review[test_idx_batch].tolist()
            
            test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                adjlists_ur, edge_metapath_indices_list_ur, test_neg_user_review_batch, device, neighbor_samples, no_masks, num_user)
            
            [neg_embedding_user, neg_embedding_review], _ = net(
                (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                
            neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
            neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
            
            neg_out = torch.bmm(neg_embedding_user, neg_embedding_review).flatten()
            test_neg_proba_list.append(torch.sigmoid(neg_out))
                
        y_proba_test = torch.cat(test_pos_proba_list + test_neg_proba_list)
        y_proba_test = y_proba_test.cpu().numpy()
        test_user_review = np.concatenate([test_pos_user_review, test_neg_user_review])
        
        build_predictions_df(category, signal, 'test', y_proba_test, test_user_review, umap, rmap)
            
        print(f'Done. Candidates can be found at {os.path.join(OUTPUT_PATH, f"ciao_{category}_{signal}")}')

if __name__ == '__main__':
    print('Starting MAGNN candidate retrieval for LlamaRec...')
    ap = argparse.ArgumentParser(description='MAGNN candidates for recommendation')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='like', help='The interaction between the user and the review. Default is like')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - node specific explicit features; ' +
                         '2 - node specific implicit features. Default is 0.')
    ap.add_argument('--sub-feats-type', type=str, default='', help='Specific type of features, e.g word2vec, tfidf.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=10, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=2, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='Ciao', help='Postfix for the saved model and result. Default is Ciao.')
    ap.add_argument('--use-pretrained', type=bool, default=False, help='Use a pre-saved model state. Default is False.')
    
    args = ap.parse_args()
    
    
    run_model_Ciao(args.category, args.signal, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, 
                   args.batch_size, args.samples)
