import wandb

import os
import sys

current_dir = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# sys.path.append('/data/ebay/data/shhirsch/ezk_packages/')
import numpy as np
import pandas as pd
# import pykrylov as krylov
import torch
import gensim
from gensim.models.keyedvectors import load_word2vec_format
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from commons import Commons
from config import hparams
from nrms import NRMS, NRMSCombined
from nrms_dataset import EvalDataset, ReviewsDataset
from utils import (verify_folder, load_checkpoint, get_logger, calc_ndcg, calc_recall_at_k, 
    calc_mrr, calc_precision_at_k, calc_hit_at_k)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(Commons.SAVED_MODELS_PATH, 'nrms')
verify_folder(checkpoint_path)
train_method_dict = {'1': 'votes', '2': 'date', '3': 'random'}
LOG_FILENAME = os.path.join(Commons.LOGS_PATH, f"nrms_{train_method_dict[hparams['train_method']]}.log")
logger = get_logger(LOG_FILENAME)


def get_eval_metrics(eval_df, mode=None):
    eval_df['order'] = eval_df['predictions'].apply(lambda x: np.argsort(x)[::-1])
    eval_df['ranked_list'] = eval_df.apply(lambda x: np.take(x['labels'], x['order']), axis=1)
    
    if hparams['use_candidates']:
        candidates_path = '/sise/bshapira-group/lilachzi/models/LlamaRec/data/preprocessed/'
        candidates_path = os.path.join(candidates_path, f"ciao_{hparams['candidates_category']}_{hparams['his_type']}")
        df = pd.read_csv(os.path.join(candidates_path, f'magnn_{mode}_{str(hparams["num_candidates"])}_candidates.csv'), 
                         converters={'y_true': eval, 'candidates': eval})
        
        df['merge_list'] = df.apply(
            lambda r: list(r['y_true'].values())[len(r['candidates']):] if len(r['y_true']) > len(r['candidates']) else [], axis=1
        )
        eval_df['ranked_list'] = eval_df['ranked_list'].apply(list) + df['merge_list']
    
    eval_df['NDCG@5'] = eval_df['ranked_list'].apply(lambda x: calc_ndcg(x[:5]) if x is not None else None)
    eval_df['MRR'] = eval_df['ranked_list'].apply(lambda x: calc_mrr(x))
    eval_df['Recall@1'] = eval_df.apply(lambda x: calc_recall_at_k(x['ranked_list'], x['ranked_list'], k=1), axis=1)
    eval_df['Recall@3'] = eval_df.apply(lambda x: calc_recall_at_k(x['ranked_list'], x['ranked_list'], k=3), axis=1)
    eval_df['Recall@5'] = eval_df.apply(lambda x: calc_recall_at_k(x['ranked_list'], x['ranked_list'], k=5), axis=1)
    eval_df['Precision@1'] = eval_df.apply(lambda x: calc_precision_at_k(x['ranked_list'], k=1), axis=1)
    eval_df['Precision@3'] = eval_df.apply(lambda x: calc_precision_at_k(x['ranked_list'], k=3), axis=1)
    eval_df['Precision@5'] = eval_df.apply(lambda x: calc_precision_at_k(x['ranked_list'], k=5), axis=1)
    eval_df['Hit@1'] = eval_df.apply(lambda x: calc_hit_at_k(x['ranked_list'], k=1), axis=1)
    eval_df['Hit@3'] = eval_df.apply(lambda x: calc_hit_at_k(x['ranked_list'], k=3), axis=1)
    eval_df['Hit@5'] = eval_df.apply(lambda x: calc_hit_at_k(x['ranked_list'], k=5), axis=1)
    ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, hit_1, hit_3, hit_5 = eval_df[
        ['NDCG@5', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Precision@1', 'Precision@3', 'Precision@5',
         'Hit@1', 'Hit@3', 'Hit@5']].mean().values

    return ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, hit_1, hit_3, hit_5


def evaluate(run_name, category=None):
    wandb.init(project='personalized_reviews', name=run_name)
    
    val_results_path = os.path.join(Commons.MODELS_PATH,
                                    f"{category.replace(' & ', '_')}_nrms_{train_method_dict[hparams['train_method']]}_val.csv")
    df_val = pd.read_csv(val_results_path)
    df_val = df_val[df_val['signal'] == hparams['his_type']]
    best_embedding = 'w2v'  # df_val.loc[best_result, 'embedding']
    # best_embedding = 'llm'  # df_val.loc[best_result, 'embedding']
    df_val = df_val[df_val['embedding'] == best_embedding]
    best_result = df_val['NDCG@5'].idxmax()
    best_batch_size = df_val.loc[best_result, 'batch_size']
    best_learning_rate = df_val.loc[best_result, 'learning_rate']
    best_weight_decay = df_val.loc[best_result, 'weight_decay']
    best_dropout = df_val.loc[best_result, 'dropout']
    his_size = df_val.loc[best_result, 'his_size']
    combined = df_val.loc[best_result, 'signal'] == 'combined'
    
    hparams['batch_size'] = best_batch_size
    hparams['lr'] = best_learning_rate
    hparams['weight_decay'] = best_weight_decay
    hparams['dropout'] = best_dropout
    print(f'DEBUG: {hparams}')

    if best_embedding == 'glove':
        embedding_model = load_word2vec_format(hparams['pretrained_model'][best_embedding], no_header=True)
    elif best_embedding == 'w2v':
        embedding_model = gensim.models.KeyedVectors.load(hparams['pretrained_model'][best_embedding])
    elif best_embedding == 'llm':
        embedding_model = AutoTokenizer.from_pretrained(
            '/sise/home/lilachzi/pretrained_models/deberta-v3-small', 
            cache_dir='/sise/home/lilachzi/pretrained_models',
            use_fast=False, model_max_length=512
        )
    
    if best_embedding == 'llm':
        reviews_ds = ReviewsDataset(embedding_model, hparams['llm_summary'])
    else:
        reviews_ds = ReviewsDataset(embedding_model, hparams['llm_summary'])
    print(f"Load test set")
    if best_embedding == 'llm':
        test_ds = EvalDataset(reviews_ds, split='test', combined=combined, category=category)
    else:
        test_ds = EvalDataset(reviews_ds, split='test', combined=combined, category=category)
    print(f"Test size {test_ds.__len__()}")
    test_dataloader = DataLoader(test_ds, batch_size=1)

    if best_embedding == 'llm':
        model = NRMSCombined(hparams, None) if combined else NRMS(hparams, None)
    else:
        model = NRMSCombined(hparams, torch.tensor(embedding_model.vectors)) if combined else NRMS(hparams, torch.tensor(embedding_model.vectors))
    if best_weight_decay < 1e-4:
        best_weight_decay = "%.5f" % best_weight_decay
    if best_learning_rate < 1e-4:
        best_learning_rate = "%.5f" % best_learning_rate
    model_name = f"{category.replace(' & ', '_')}_nrms_{train_method_dict[hparams['train_method']]}_{his_size}_{hparams['his_type']}_bs_{str(best_batch_size)}_lr_" \
                 f"{str(best_learning_rate).split('.')[1]}_wd_{str(best_weight_decay).split('.')[1]}_d_" \
                 f"{str(best_dropout).split('.')[1]}_e_{best_embedding}_{hparams['llm_summary']}"
    model = load_checkpoint(checkpoint_path, model_name, device, model)

    predictions = []
    labels = []
    model = model.eval()
    with torch.no_grad():
        for batch_idx, test_batch in enumerate(test_dataloader):
            if batch_idx % 100 == 0:
                print(f'DEBUG: Iteration {batch_idx}')
            
            test_batch = tuple(t.to(device) for t in test_batch)
            cands_label = test_batch[-1]
            logits = model(*test_batch[:-1], is_eval=True)
            predictions.append(logits.cpu().numpy()[0])
            labels.append(cands_label.cpu().numpy()[0])

    df_preds = pd.DataFrame({'predictions': predictions, 'labels': labels})
    test_votes_file = os.path.join(Commons.CSVS_PATH, 'test_set.csv')
    test_df = pd.read_csv(test_votes_file)
    if category:
        test_df = test_df[test_df['category'] == category]

    test_results = pd.concat([test_df.reset_index(drop=True), df_preds], axis=1)
    test_results = test_results[['product_id', 'voter_id', 'labels', 'predictions']]
    preds_file_name = f"{category.replace(' & ', '_')}_{model_name}_preds.csv"
    test_results.to_csv(os.path.join(Commons.PREDICTIONS_PATH, preds_file_name), index=False)
    ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, \
        hit_1, hit_3, hit_5 = get_eval_metrics(test_results)
    
    # wandb.log({
    #     f'scores/nrms/nDCG': ndcg,
    #     f'scores/nrms/Recall@1': recall_1, 
    #     f'scores/nrms/Recall@3': recall_3, 
    #     f'scores/nrms/Recall@5': recall_5
    # })
    
    results = pd.DataFrame([[category, model_name, ndcg, mrr, recall_1, recall_3, recall_5, precision_1, precision_3, precision_5, hit_1, hit_3, hit_5]],
                           columns=['Category', 'Model', 'NDCG@5', 'MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Precision@1', 'Precision@3', 'Precision@5',
                                    'Hit@1', 'Hit@3', 'Hit@5'])

    results_path = os.path.join(Commons.MODELS_PATH, 'model_results.csv')
    if os.path.exists(results_path):
        results.to_csv(results_path, index=False, mode='a', header=False)
    else:
        results.to_csv(results_path, index=False)
        
    # removed_models = []
    # # remove other models to save space on disk
    # for idx, row in df_val.loc[df_val.index != best_result].iterrows():
    #     if row['weight_decay'] < 1e-4:
    #         weight_decay = "%.5f" % row['weight_decay']
    #     else:
    #         weight_decay = row['weight_decay']
    #     if row['learning_rate'] < 1e-4:
    #         learning_rate = "%.5f" % row['learning_rate']
    #     else:
    #         learning_rate = row['learning_rate']
            
    #     model_name_to_remove = f"{category}_nrms_{train_method_dict[hparams['train_method']]}_{row['his_size']}_{row['signal']}_bs_{str(row['batch_size'])}_lr_" \
    #              f"{str(learning_rate).split('.')[1]}_wd_{str(weight_decay).split('.')[1]}_d_" \
    #              f"{str(row['dropout']).split('.')[1]}_e_{best_embedding}_{hparams['llm_summary']}.pt"
    #     if model_name_to_remove in removed_models:
    #         continue
        
    #     print(f'removing model: {model_name_to_remove}')
    #     os.remove(os.path.join(checkpoint_path, model_name_to_remove))
    #     removed_models.append(model_name_to_remove)


def evaluating_wrapper(category=None):
    print("submitting experiment")
    project_name = "personalization"
    evaluate_model_task = krylov.Task(evaluate,
                                      docker_image="ecr.vip.ebayc3.com/glavee/ez_krylov:latest",
                                      args=(category,)
                                      )
    evaluate_model_task.run_on_gpu(1, model="v100g3")
    # train_model_task.add_cpu(4)
    evaluate_model_task.add_memory(32)
    evaluate_model_task.add_environment_variable("PYTHONPATH", "/data/ebay/data/shhirsch/personalization/")

    experiment_id = krylov.Session().submit_experiment(evaluate_model_task,
                                                       project=project_name,
                                                       experiment_name=f"evaluate_nrms_{train_method_dict[hparams['train_method']]}_"
                                                                       f"{hparams['his_type']}",
                                                       labels=[
                                                           "embedding=glove",
                                                       ]
                                                       )
    print("https://94.aihub.krylov.vip.ebay.com/projects/" + project_name + "/experiments/" + experiment_id)


if __name__ == '__main__':
    # evaluating_wrapper(category='Beauty')
    run_name = f'nrms_votes_{hparams["his_type"]}_eval'
    evaluate(run_name, category='Food & Drink')
