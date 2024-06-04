import argparse
import gensim
import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import hparams
from nrms import NRMS, NRMSCombined
from nrms_dataset import ReviewsDataset, TrainDataset, EvalDataset
from utils import load_checkpoint

BASE_PATH = '/sise/bshapira-group/lilachzi/models'
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'nrms', 'saved_models', 'nrms')
CSVS_PATH = '/sise/bshapira-group/lilachzi/csvs'
MODELS_PATH = os.path.join(BASE_PATH, 'nrms', 'models')
OUTPUT_PATH = os.path.join(BASE_PATH, 'LlamaRec/data/preprocessed')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_predictions_df(category, signal, mode, predictions, labels, num_candidates):
    if signal == 'combined':
        signal = 'both'

    df_preds = pd.DataFrame({'predictions': predictions, 'labels': labels})
    df = pd.read_csv(os.path.join(CSVS_PATH, f'{mode}_set.csv'), converters={'y_true': eval})
    df = df[df['category'] == category]
    
    results = pd.concat([df.reset_index(drop=True), df_preds], axis=1)
    
    results['order'] = results['predictions'].apply(lambda x: np.argsort(x)[::-1])
    results['candidates'] = results['y_true'].apply(lambda x: list(x.keys()))
    results['candidates'] = results.apply(lambda x: np.take(x['candidates'], x['order']), axis=1)
    results['candidates'] = results['candidates'].apply(lambda lst: lst[:num_candidates])
    results['candidates'] = results.apply(lambda r: {rid: r['y_true'][rid] for rid in r['candidates']}, axis=1)
    
    results = results[['product_id', 'voter_id', 'candidates', 'y_true']]

    results.to_csv(os.path.join(OUTPUT_PATH, f'ciao_{category.replace(" & ", "_")}_{signal}', f'nrms_{mode}_{str(num_candidates)}_candidates.csv'), index=False)

def run_model_nrms(category, signal, num_candidates):
    val_results_path = os.path.join(MODELS_PATH, f"{category.replace(' & ', '_')}_nrms_votes_val.csv")
    df_val = pd.read_csv(val_results_path)
    df_val = df_val[df_val['signal'] == signal]
    
    best_result = df_val['NDCG@5'].idxmax()
    best_embedding = df_val.loc[best_result, 'embedding']
    best_summary = None if np.isnan(df_val.loc[best_result, 'llm_summary']) else df_val.loc[best_result, 'llm_summary']
    best_batch_size = df_val.loc[best_result, 'batch_size']
    best_learning_rate = df_val.loc[best_result, 'learning_rate']
    best_weight_decay = df_val.loc[best_result, 'weight_decay']
    best_dropout = df_val.loc[best_result, 'dropout']
    his_size = df_val.loc[best_result, 'his_size']
    combined = df_val.loc[best_result, 'signal'] == 'combined'
    
    hparams['embedding_type'] = best_embedding
    hparams['llm_summary'] = best_summary
    
    if best_embedding == 'w2v':
        embedding_model = gensim.models.KeyedVectors.load(hparams['pretrained_model'][best_embedding])
    elif best_embedding == 'llm':
        embedding_model = AutoTokenizer.from_pretrained(
            'google-bert/bert-base-uncased', model_max_length=hparams['review_size']
        )
        llm_model = AutoModel.from_pretrained('google-bert/bert-base-uncased')
        
    print("Load datasets")
    reviews_ds = ReviewsDataset(embedding_model, best_summary, best_embedding)
    valid_ds = EvalDataset(reviews_ds, split='valid', combined=combined, category=category)
    test_ds = EvalDataset(reviews_ds, split='test', combined=combined, category=category)
    valid_dataloader = DataLoader(valid_ds, batch_size=1)
    test_dataloader = DataLoader(test_ds, batch_size=1)
    
    if combined:
        model = NRMSCombined(hparams, torch.tensor(embedding_model.vectors))
    else:
        if best_embedding == 'llm':
            model = NRMS(hparams, llm_model=llm_model)
        else:
            model = NRMS(hparams, torch.tensor(embedding_model.vectors))
    
    if best_weight_decay < 1e-4:
        best_weight_decay = "%.5f" % best_weight_decay
    if best_learning_rate < 1e-4:
        best_learning_rate = "%.5f" % best_learning_rate
    model_name = f"{category.replace(' & ', '_')}_nrms_votes_{his_size}_{signal}_bs_{str(best_batch_size)}_lr_" \
                 f"{str(best_learning_rate).split('.')[1]}_wd_{str(best_weight_decay).split('.')[1]}_d_" \
                 f"{str(best_dropout).split('.')[1]}_e_{best_embedding}_{hparams['llm_summary']}"
    model = load_checkpoint(CHECKPOINT_PATH, model_name, device, model)
    
    val_predictions = []
    val_labels = []
    test_predictions = []
    test_labels = []
    model = model.eval()
    with torch.no_grad():
        for batch_idx, valid_batch in tqdm(enumerate(valid_dataloader), total=valid_dataloader.__len__(), desc='Validation candidates'):
            valid_batch = tuple(t.to(device) for t in valid_batch)
            if combined:
                cands_label = valid_batch[-1]
                logits = model(*valid_batch[:-1], is_eval=True)
            else:
                history_attn, cand_attn = None, None
                if len(valid_batch) > 3:
                    # embedding type = llm
                    history_attn = valid_batch[-2]
                    cand_attn = valid_batch[-1]
                cands_label = valid_batch[2]
                logits = model(*valid_batch[:2], is_eval=True, history_attn=history_attn, candidates_attn=cand_attn)
            val_predictions.append(logits.cpu().numpy()[0])
            val_labels.append(cands_label.cpu().numpy()[0])
            
        build_predictions_df(category, signal, 'valid', val_predictions, val_labels, num_candidates)
        
        for batch_idx, test_batch in tqdm(enumerate(test_dataloader), total=test_dataloader.__len__(), desc='Test candidates'):
            test_batch = tuple(t.to(device) for t in test_batch)
            if combined:
                cands_label = test_batch[-1]
                logits = model(*test_batch[:-1], is_eval=True)
            else:
                history_attn, cand_attn = None, None
                if len(test_batch) > 3:
                    # embedding type = llm
                    history_attn = test_batch[-2]
                    cand_attn = test_batch[-1]
                cands_label = test_batch[2]
                logits = model(*test_batch[:2], is_eval=True, history_attn=history_attn, candidates_attn=cand_attn)
            test_predictions.append(logits.cpu().numpy()[0])
            test_labels.append(cands_label.cpu().numpy()[0])
            
        build_predictions_df(category, signal, 'test', test_predictions, test_labels, num_candidates)
        
    print(f'Done. Candidates can be found at {os.path.join(OUTPUT_PATH, f"ciao_{category}_{signal}")}')

if __name__ == '__main__':
    print('Starting NRMS candidate retrieval for LlamaRec...')
    ap = argparse.ArgumentParser(description='NRMS candidates for recommendation')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='like', help='The interaction between the user and the review. Default is like')
    ap.add_argument('--num_candidates', type=int, help='Number of candidates to output.')
    
    args = ap.parse_args()
    
    run_model_nrms(args.category, args.signal, args.num_candidates)
