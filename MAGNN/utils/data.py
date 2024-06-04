import numpy as np
import scipy
import pickle
import os
import time
import torch

DATA_PATH = '/sise/bshapira-group/lilachzi/models/magnn/data'


def load_ciao_data(category, signal, feats_type, sub_feats_type = ''):
    print('Loading data...')
    start_time = time.time()
    path = os.path.join(DATA_PATH, f'preprocessed/Ciao{category.replace(" & ", "_")}_{signal}_processed')
    features_path = os.path.join(DATA_PATH, f'preprocessed/Ciao{category.replace(" & ", "_")}_like_processed')
    
    adj_lists = []
    index_lists = []
    for root, dirs, files in os.walk(path):
        for i, dir in enumerate(dirs):
            adj_lists.append([])
            adj_files = sorted([f for f in os.listdir(os.path.join(path, dir)) if f.endswith('.adjlist')])
            for file in adj_files:
                with open(os.path.join(path, dir, file), 'r') as f:
                    adjlist = [line.strip() for line in f]
                    adj_lists[i].append(adjlist)
                
            index_lists.append([])
            index_files = sorted([f for f in os.listdir(os.path.join(path, dir)) if f.endswith('.pickle')])
            for file in index_files:
                with open(os.path.join(path, dir, file), 'rb') as f:
                    idx = pickle.load(f)
                    index_lists[i].append(idx)
                    
    adjM = scipy.sparse.load_npz(os.path.join(path, 'adj_mat.npz'))
    type_mask = np.load(os.path.join(path, 'node_types.npy'))
    train_val_test_pos_user_review = np.load(os.path.join(path, 'train_val_test_pos_user_review.npz'))
    train_val_test_neg_user_review = np.load(os.path.join(path, 'train_val_test_neg_user_review.npz'))
    
    features = []
    if feats_type == 1:
        features = np.load(os.path.join(features_path, f'{sub_feats_type}explicit_features.npz'))
    elif feats_type == 2:
        features = np.load(os.path.join(features_path, f'{sub_feats_type}implicit_features.npz'))
    
    print(f'Data loading finished after {time.time() - start_time:.2f} seconds')

    return adj_lists, index_lists, adjM, type_mask, train_val_test_pos_user_review, \
            train_val_test_neg_user_review, features
            
def load_mappings(category, signal):
    path = os.path.join(DATA_PATH, f'preprocessed/Ciao{category.replace(" & ", "_")}_{signal}_processed')
    
    with open(os.path.join(path, 'mappings.pickle'), 'rb') as f:
        mappings = pickle.load(f)
        
    return mappings['umap'], mappings['rmap'], mappings['pmap']
            
def load_features_info(category, signal, feats_type, sub_feats_type = ''):
    path = os.path.join(DATA_PATH, f'preprocessed/Ciao{category.replace(" & ", "_")}_{signal}_processed')
    features_path = os.path.join(DATA_PATH, f'preprocessed/Ciao{category.replace(" & ", "_")}_like_processed')
    
    type_mask = np.load(os.path.join(path, 'node_types.npy'))
    
    features = []
    if feats_type == 1:
        features = np.load(os.path.join(features_path, f'/{sub_feats_type}explicit_features.npz'))
    elif feats_type == 2:
        features = np.load(os.path.join(features_path, f'/{sub_feats_type}implicit_features.npz'))
        
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(3):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to('cuda'))
    else:
        features_list = [torch.FloatTensor(features_type).to('cuda') for features_type in features.values()]
        in_dims = [features.shape[1] for features in features_list]
        
    return in_dims

def get_metapaths_info(signal):
    if signal == 'like':
        # 0: user liked review, 1: review liked by user, 2: review under product
        # 3: product contains review, 4: user liked review under product, 5: product contains review liked by user
        etypes_lists = [[[0, 1], [0, 2, 3, 1], [4, 5]],
                        [[1, 0], [2, 5, 4, 3], [2, 3]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 6
        use_masks = [[True, True, False],
                    [True, True, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    elif signal == 'write':
        # 0: user wrote review, 1: review written by user, 2: review under product
        # 3: product contains review, 4: user wrote review under product, 5: product contains review written by user
        etypes_lists = [[[0, 2, 3, 1], [4, 5]],
                        [[1, 0], [2, 5, 4, 3], [2, 3]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 6
        use_masks = [[True, False],
                    [True, True, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    elif signal == 'both':
        # 0: user liked review, 1: review liked by user, 2: review under product, 3: product contains review
        # 4: user wrote review under product, 5: product contains review written by user
        # 6: user written review, 7: review written by user
        etypes_lists = [[[0, 1], [0, 2, 3, 1], [4, 5], [6, 2, 3, 7]],
                        [[1, 0], [2, 5, 4, 3], [2, 3], [7, 6]]]
        num_metapaths_list = [len(lst) for lst in etypes_lists]
        num_edge_type = 8
        use_masks = [[True, True, True, False],
                    [True, True, False, False]]
        no_masks = [[False] * mp for mp in num_metapaths_list]
    else:
        raise Exception('wrong signal was given.')
        
    return etypes_lists, num_metapaths_list, num_edge_type, use_masks, no_masks