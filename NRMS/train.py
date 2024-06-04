import wandb

import os
import sys

current_dir = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/data/ebay/data/shhirsch/ezk_packages/')
# import ez_krylov
# ez_krylov.ensure_package(package_name='gensim')
# ez_krylov.ensure_package(package_name='nltk')
# import pykrylov as krylov
import pandas as pd
import itertools
import torch
import gensim
from gensim.models.keyedvectors import load_word2vec_format
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from commons import Commons
from config import hparams
from nrms import NRMS, NRMSCombined
from nrms_dataset import ReviewsDataset, TrainDataset, EvalDataset
from evaluate import get_eval_metrics
from utils import verify_folder, save_checkpoint, load_checkpoint, get_logger

train_method_dict = {'1': 'votes', '2': 'date', '3': 'random'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(Commons.SAVED_MODELS_PATH, 'nrms')
LOG_FILENAME = os.path.join(Commons.LOGS_PATH, f"nrms_{train_method_dict[hparams['train_method']]}.log")
logger = get_logger(LOG_FILENAME)
verify_folder(checkpoint_path)


def eval_model(model, valid_dataloader, return_loss=True):
    model = model.eval()
    val_loss = []
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for valid_batch in valid_dataloader:
            valid_batch = tuple(t.to(device) for t in valid_batch)
            if hparams['his_type'] == 'combined':
                cands_label = valid_batch[-1]
                if return_loss:
                    loss, logits = model(*valid_batch, is_eval=True)
                    val_loss.append(loss.item())
                else:
                    logits = model(*valid_batch[:-1], is_eval=True)
            else:
                history_attn, cand_attn = None, None
                if len(valid_batch) > 3:
                    # embedding type = llm
                    history_attn = valid_batch[-2]
                    cand_attn = valid_batch[-1]
                cands_label = valid_batch[2]
                if return_loss:
                    loss, logits = model(*valid_batch[:3], is_eval=True, history_attn=history_attn, candidates_attn=cand_attn)
                    val_loss.append(loss.item())
                else:
                    logits = model(*valid_batch[:2], is_eval=True, history_attn=history_attn, candidates_attn=cand_attn)

            val_predictions.append(logits.cpu().numpy()[0])
            val_labels.append(cands_label.cpu().numpy()[0])

    df_preds = pd.DataFrame({'predictions': val_predictions, 'labels': val_labels})
    ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, \
        hit_1, hit_3, hit_5 = get_eval_metrics(df_preds, 'valid')
    if return_loss:
        val_loss_mean = torch.mean(torch.tensor(val_loss))
    else:
        val_loss_mean = None

    return val_loss_mean, ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, \
        hit_1, hit_3, hit_5


def train(batch_size,
          learning_rate,
          weight_decay,
          dropout,
          embedding_type,
          category=None,
          num_epochs=10,
          best_valid_loss=float("Inf"),
          patience=2,
          min_delta=0.0001,
          ):
    batch_size = int(batch_size)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    dropout = float(dropout)
    hparams['dropout'] = dropout
    combined = hparams['his_type'] == 'combined'
    val_results_path = os.path.join(Commons.MODELS_PATH,
                                    f"{category.replace(' & ', '_')}_nrms_{train_method_dict[hparams['train_method']]}_val.csv")
    if hparams['use_candidates']:
        val_results_path = os.path.join(Commons.MODELS_PATH,
                                    f"{category.replace(' & ', '_')}_nrms_{train_method_dict[hparams['train_method']]}_magnn_candidates_val.csv")

    if embedding_type == 'glove':
        embedding_model = load_word2vec_format(hparams['pretrained_model'][embedding_type], no_header=True)
    elif embedding_type == 'w2v':
        embedding_model = gensim.models.KeyedVectors.load(hparams['pretrained_model'][embedding_type])
    elif embedding_type == 'llm':
        embedding_model = AutoTokenizer.from_pretrained(
            'google-bert/bert-base-uncased', model_max_length=hparams['review_size']
        )
        llm_model = AutoModel.from_pretrained('google-bert/bert-base-uncased')

    print("Load datasets")
    reviews_ds = ReviewsDataset(embedding_model, hparams['llm_summary'], hparams['embedding_type'])
    train_ds = TrainDataset(reviews_ds, combined=combined, category=category)
    valid_ds = EvalDataset(reviews_ds, split='valid', combined=combined, category=category)
    print("Finish load datasets")
    print(f"Train size {train_ds.__len__()}")
    print(f"Valid size {valid_ds.__len__()}")

    if combined:
        model = NRMSCombined(hparams, torch.tensor(embedding_model.vectors))
    else:
        if embedding_type == 'llm':
            model = NRMS(hparams, llm_model=llm_model)
        else:
            model = NRMS(hparams, torch.tensor(embedding_model.vectors))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=1)

    if weight_decay < 1e-4:
        weight_decay = "%.5f" % weight_decay
    if learning_rate < 1e-4:
        learning_rate = "%.5f" % learning_rate
    hparams['his_size'] = hparams['like_his_size'] + hparams['write_his_size'] if combined else hparams['his_size']
    model_name = f"{category.replace(' & ', '_')}_nrms_{train_method_dict[hparams['train_method']]}_{hparams['his_size']}_{hparams['his_type']}_bs_{str(batch_size)}_lr_" \
                 f"{str(learning_rate).split('.')[1]}_wd_{str(weight_decay).split('.')[1]}_d_" \
                 f"{str(dropout).split('.')[1]}_e_{embedding_type}_{hparams['llm_summary']}"
    if hparams['use_candidates']:
        model_name = f'{model_name}_magnn_candidates_{str(hparams["num_candidates"])}'

    wandb.init(
        # set the wandb project where this run will be logged
        project='personalized_reviews',
        name=model_name,
        
        # track hyperparameters and run metadata
        config = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay, 
            'dropout_rate': dropout, 
            'batch_size': batch_size,
            'model_name': f'{category.replace(" & ", "_")}_nrms'
        }
    )
    
    print(f'DEBUG: run parameters - {hparams}')
    
    # Uncomment if needed
    # print(f'Loading pretrained model: {model_name}')
    # model = load_checkpoint(checkpoint_path, model_name, device, model)

    train_losses, val_losses = [], []
    termination = False
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}')
        # TRAINING LOOP
        model = model.train()
        train_loss = []
        for train_batch in train_dataloader:
            train_batch = tuple(t.to(device) for t in train_batch)
            if hparams['embedding_type'] == 'llm':
                history, cand, labels, history_attn, cand_attn = train_batch
                loss, logits = model(history, cand, labels, history_attn=history_attn, candidates_attn=cand_attn)
            else:
                loss, logits = model(*train_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            wandb.log({'train_loss': loss.item()})

        train_loss_mean = torch.mean(torch.tensor(train_loss))
        train_losses.append(train_loss_mean)

        # VALIDATION LOOP
        val_loss_mean, ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, \
            hit_1, hit_3, hit_5 = eval_model(model, valid_dataloader)
        val_losses.append(val_loss_mean)
        wandb.log({'val_loss': val_loss_mean})

        if best_valid_loss - min_delta <= val_loss_mean:
            patience -= 1
            if patience == 0:
                termination = True
        else:
            patience = 2
            best_valid_loss = val_loss_mean
            save_checkpoint(checkpoint_path, model_name, model)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid NDCG: {:.4f}, Valid MRR: {:.4f}, Valid Recall@5: {:.4f}'
                    .format(epoch + 1, num_epochs, train_loss_mean, val_loss_mean, ndcg, mrr, recall_5))

        if termination:
            print('Early stopping - stop training!')
            break

    if combined:
        best_model = NRMSCombined(hparams, torch.tensor(embedding_model.vectors))
    else:
        if embedding_type == 'llm':
            best_model = NRMS(hparams, None)
        else:
            best_model = NRMS(hparams, torch.tensor(embedding_model.vectors))
    best_model = load_checkpoint(checkpoint_path, model_name, device, best_model)
    _, ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, \
        hit_1, hit_3, hit_5 = eval_model(best_model, valid_dataloader, return_loss=False)
    df_val = pd.DataFrame(
        data=[[hparams['his_type'], num_epochs, hparams['his_size'], batch_size, learning_rate, weight_decay, dropout, embedding_type, hparams['llm_summary'],
               ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, hit_1, hit_3, hit_5]],
        columns=['signal', 'epochs', 'his_size', 'batch_size', 'learning_rate', 'weight_decay', 'dropout', 'embedding', 'llm_summary',
                 'NDCG@5', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Precision@1', 'Precision@3', 'Precision@5',
                 'Hit@1', 'Hit@3', 'Hit@5'])
    if os.path.exists(val_results_path):
        df_val.to_csv(val_results_path, index=False, mode='a', header=False)
    else:
        df_val.to_csv(val_results_path, index=False)

    print('finish training')


if __name__ == '__main__':
    train(batch_size=hparams['batch_size'],
          learning_rate=hparams['lr'],
          weight_decay=hparams['weight_decay'],
          dropout=hparams['dropout'],
          embedding_type=hparams['embedding_type'],
          category='Internet')
