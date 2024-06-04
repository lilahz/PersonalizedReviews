import time
import argparse
import os
import pickle
import wandb

import torch
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
from utils.pytorchtools import EarlyStopping
from utils.data import load_ciao_data, get_metapaths_info
from utils.tools import index_generator, parse_minibatch_Ciao
from utils.eval import evaluate_link_prediction, evaluate_cosine_similarity
from model import MAGNN_lp

BASE_PATH = '/sise/bshapira-group/lilachzi'
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'models/magnn/checkpoint')
RESULTS_PATH = os.path.join(BASE_PATH, 'models/magnn/data/results')

# Params
num_ntype = 3
# Beauty
# num_user = 4417
# num_review = 14091
# Games
num_user = 4269
num_review = 8919
# Food & Drink
# num_user = 4410
# num_review = 12084
# DVDs
# num_user = 5189
# num_review = 27964
# Internet
# num_user = 4965
# num_review = 11737


def run_model_Ciao(category, signal, feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix, **kwargs):
    dropout_rate = kwargs['dropout_rate']
    lr = kwargs['lr']
    weight_decay = kwargs['weight_decay']
    sub_feats_type = kwargs['sub_feats_type']
    
    save_postfix = kwargs['run_name']
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
        
    train_pos_user_review = train_val_test_pos_user_review['train_pos_user_review']
    val_pos_user_review = train_val_test_pos_user_review['val_pos_user_review']
    test_pos_user_review = train_val_test_pos_user_review['test_pos_user_review']
    train_neg_user_review = train_val_test_neg_user_review['train_neg_user_review']
    val_neg_user_review = train_val_test_neg_user_review['val_neg_user_review']
    test_neg_user_review = train_val_test_neg_user_review['test_neg_user_review']
    
    etypes_lists, num_metapaths_list, num_edge_type, use_masks, no_masks = get_metapaths_info(signal)
    
    for _ in range(repeat):
        net = MAGNN_lp(
            num_metapaths_list, num_edge_type, etypes_lists, in_dims, hidden_dim, hidden_dim, 
            num_heads, attn_vec_dim, rnn_type, dropout_rate
        )
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        
        if kwargs['use_pretrained']:
            print('Loading pre-saved version of model...')
            net.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')))

        # training loop
        print(f'DEBUG: starting train on {len(train_pos_user_review)} samples.')
        net.train()
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, save_path=os.path.join(CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')
        )
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_user_review))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_user_review), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_user_review_batch = train_pos_user_review[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_user_review), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_user_review_batch = train_neg_user_review[train_neg_idx_batch].tolist()

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, train_pos_user_review_batch, device, neighbor_samples, use_masks, num_user)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, train_neg_user_review_batch, device, neighbor_samples, no_masks, num_user)

                t1 = time.time()
                dur1.append(t1 - t0)

                [pos_embedding_user, pos_embedding_review], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_user, neg_embedding_review], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
                
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_user, pos_embedding_review)
                neg_out = -torch.bmm(neg_embedding_user, neg_embedding_review)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    wandb.log({'train_loss': train_loss.item()})
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_user_review_batch = val_pos_user_review[val_idx_batch].tolist()
                    val_neg_user_review_batch = val_neg_user_review[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                        adjlists_ur, edge_metapath_indices_list_ur, val_pos_user_review_batch, device, neighbor_samples, no_masks, num_user)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                        adjlists_ur, edge_metapath_indices_list_ur, val_neg_user_review_batch, device, neighbor_samples, no_masks, num_user)

                    [pos_embedding_user, pos_embedding_review], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_user, neg_embedding_review], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    
                    pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)
                    neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_review)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_review)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                    
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            wandb.log({'val_loss': val_loss.item()})
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        pos_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_user_review), shuffle=False)
        neg_test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_neg_user_review), shuffle=False)
        net.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, f'checkpoint_{save_postfix}.pt')))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        users_embeddings = defaultdict(list)
        reviews_embeddings = defaultdict(list)
        with torch.no_grad():
            for iteration in range(pos_test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = pos_test_idx_generator.next()
                test_pos_user_review_batch = test_pos_user_review[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, test_pos_user_review_batch, device, neighbor_samples, no_masks, num_user)

                [pos_embedding_user, pos_embedding_review], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                
                for i, (user_id, review_id) in enumerate(test_pos_user_review_batch):
                    users_embeddings[user_id].append(pos_embedding_user[i])
                    reviews_embeddings[review_id].append(pos_embedding_review[i])
                
                pos_embedding_user = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_review = pos_embedding_review.view(-1, pos_embedding_review.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_review).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                
            for iteration in range(neg_test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = neg_test_idx_generator.next()
                test_neg_user_review_batch = test_neg_user_review[test_idx_batch].tolist()
                
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_Ciao(
                    adjlists_ur, edge_metapath_indices_list_ur, test_neg_user_review_batch, device, neighbor_samples, no_masks, num_user)
                
                [neg_embedding_user, neg_embedding_review], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                
                for i, (user_id, review_id) in enumerate(test_neg_user_review_batch):
                    users_embeddings[user_id].append(neg_embedding_user[i])
                    reviews_embeddings[review_id].append(neg_embedding_review[i])
                    
                neg_embedding_user = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_review = neg_embedding_review.view(-1, neg_embedding_review.shape[1], 1)
                
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_review).flatten()
                neg_proba_list.append(torch.sigmoid(neg_out))
                
            users_embeddings = {user_id: torch.mean(torch.stack(embeddings), dim=0, keepdim=True).cpu().numpy() for user_id, embeddings in users_embeddings.items()}
            reviews_embeddings = {review_id: torch.mean(torch.stack(embeddings), dim=0, keepdim=True).cpu().numpy() for review_id, embeddings in reviews_embeddings.items()}
            
            with open(os.path.join(RESULTS_PATH, f'Ciao{category.replace(" & ", "_")}/user_embds_{save_postfix}.pkl'), 'wb') as f:
                pickle.dump(users_embeddings, f)
            with open(os.path.join(RESULTS_PATH, f'Ciao{category.replace(" & ", "_")}/review_embds_{save_postfix}.pkl'), 'wb') as f:
                pickle.dump(reviews_embeddings, f)
                
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
            with open(os.path.join(RESULTS_PATH, f'Ciao{category.replace(" & ", "_")}/y_proba_test_{save_postfix}.npy'), 'wb') as f:
                np.save(f, y_proba_test)
        
        print('----------------------------------------------------------------')
        print('Link Prediction Test')
        evaluate_link_prediction(test_pos_user_review, test_neg_user_review, y_proba_test, kwargs['run_name'], category)


if __name__ == '__main__':
    print('Starting MAGNN testing for Ciao...')
    ap = argparse.ArgumentParser(description='MAGNN testing for the recommendation dataset')
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
    
    lr = 0.0001
    weight_decay = 0
    dropout_rate = 0.5
                
    run_name = f'Ciao{args.category}_{args.signal}_{args.sub_feats_type}{args.feats_type}_{lr}_{weight_decay}_{dropout_rate}_{args.batch_size}'
    wandb.init(
        # set the wandb project where this run will be logged
        project='personalized_reviews',
        name=run_name,
        
        # track hyperparameters and run metadata
        config = {
            'learning_rate': lr,
            'weight_decay': weight_decay, 
            'dropout_rate': dropout_rate, 
            'batch_size': args.batch_size,
            'features': args.feats_type,
            'model_name': f'Ciao{args.category}'
        }
    )
    
    run_model_Ciao(args.category, args.signal, args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                    args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix, lr=lr, weight_decay=weight_decay,
                    dropout_rate=dropout_rate, run_name=run_name, use_pretrained=args.use_pretrained, sub_feats_type=args.sub_feats_type)
