import argparse
import os
import pandas as pd
import random

from langchain.llms import LlamaCpp
from tqdm import tqdm

CSVS_PATH = '/sise/bshapira-group/lilachzi/csvs'
MODELS_PATH = '/sise/bshapira-group/lilachzi/models'


def load_data():
    reviews = pd.read_csv(os.path.join(CSVS_PATH, 'ciao_4core.tsv'), sep='\t')
    votes = pd.read_csv(os.path.join(CSVS_PATH, 'votes_4core_split_60_20_20.tsv'), sep='\t')
    
    written_reviews = {}

    for _, row in tqdm(reviews.iterrows(), total=len(reviews)):
        user_id = row['user_id']

        if user_id not in written_reviews:
            written_reviews[user_id] = []
        written_reviews[user_id].append((row['review_id'], row['category'], row['product']))

    liked_reviews = {}
    positive_vote = [3, 4, 5]

    for _, row in tqdm(votes.iterrows(), total=len(votes)):
        voter_id = row['voter_id']
        split = row['split']
        
        if row['vote'] in positive_vote:
            if voter_id not in liked_reviews:
                liked_reviews[voter_id] = {}
            if split not in liked_reviews[voter_id]:
                liked_reviews[voter_id][split] = []
            liked_reviews[voter_id][split].append((row['review_id'], row['category'], row['product']))
            
    reviews['review_concat'] = reviews['clean_review'].apply(lambda t: ' '.join(t.split()[:150]))
    review_id_text_mapping = dict(zip(reviews.review_id, reviews.review_concat))
            
    return written_reviews, liked_reviews, review_id_text_mapping

def load_history(voter_id, rid2index, written_review, liked_reviews, split, history_type, category):
    history_size = 10
    
    history_write = [(i[1], i[2], rid2index[i[0]]) for i in written_review[voter_id]]
    if split in liked_reviews.get(voter_id, []):
        liked_reviews_user = liked_reviews[voter_id][split]
    else:
        liked_reviews_user = []
        
    history_like = [(i[1], i[2], rid2index[i[0]]) for i in liked_reviews_user if i[1] != category]

    if history_type == 'write':
        history = random.sample(history_write, min(history_size, len(history_write)))
    elif history_type == 'like':
        history = random.sample(history_like, min(history_size, len(history_like)))
    else:  # 'both'
        history = (list(random.sample(history_write, min(history_size//2, len(history_write))) +
                   list(random.sample(history_like, min(history_size//2, len(history_like))))))

    return history

def build_profile(llm, signal, history):
    if signal == 'like':
        past_signal = 'liked'
    elif signal == 'write':
        past_signal = 'written'
    else:
        past_signal = 'liked or wrote'
        
    prompt = f"""You are asked to describe user interests and preferences based on his/her {past_signal} reviews list, 
        Your'e given the user's past {past_signal} reviews in the format: <product category, product title> : <product review content>
        You can only response the user interests and preferences (at most 10 sentences). Don't use lists, use summary structure.
        The output should begin with the word Profile:.
        These are the {past_signal} reviews : \n {history}.
        """

    try:
        output = llm(f"<s>[INST] {prompt} [/INST]", max_tokens=-1)
    except:
        output = ''

    return output
    
def build_user_profiles(llm, category, signal, split, written_history, liked_history, review_id_text_mapping):
    profiles = []
    
    train_set = pd.read_csv(os.path.join(CSVS_PATH, f'{split}_set.csv'))
    train_set = train_set[train_set['category'] == category]
    
    temp_file = os.path.join(os.path.dirname(__file__), 'profiles_temp.csv')
    if os.path.exists(temp_file):
        temp = pd.read_csv(temp_file)
        profiles = list(temp.itertuples(index=False, name=None))
        
        train_set = train_set[~train_set['voter_id'].isin(temp['user_id'].tolist())]
    
    for idx, user in tqdm(enumerate(list(train_set['voter_id'].unique())), total=train_set['voter_id'].nunique()):
        history_info = load_history(user, review_id_text_mapping, written_history, liked_history,
                               split, signal, category)
        history = [f'<{h[0]}, {h[1]}>: <{h[2].strip()}>' for h in history_info]
        history = '\n'.join(history)
        
        profile = build_profile(llm, signal, history)
        profiles.append((user, history_info, profile))
        
        if idx % 10 == 0:
           pd.DataFrame(profiles, columns=['user_id', 'history', 'profile']).to_csv(
               os.path.join(os.path.dirname(__file__), 'profiles_temp.csv'), index=False)
        
    return pd.DataFrame(profiles, columns=['user_id', 'history', 'profile'])
    
def run(category, signal, split):
    model_path = os.path.join(MODELS_PATH, 'pretrained_models/llama-cpp-python/llama-2-7b-chat.Q4_K_S.gguf')
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=10000,
        temperature=0.5,
        max_tokens=200,
        n_gpu_layers=-1,
        n_batch=512
    )
    
    written_reviews, liked_reviews, review_id_text_mapping = load_data()
    return build_user_profiles(llm, category, signal, split, written_reviews, liked_reviews, review_id_text_mapping)
    
    
if __name__ == '__main__':
    print('Starting profile generation usign Llama CPP')
    ap = argparse.ArgumentParser(description='User profile generation')
    ap.add_argument('--category', default='Beauty', help='Name of Ciao category. Default is Beauty.')
    ap.add_argument('--signal', default='like', help='The interaction between the user and the review. Default is like')
    ap.add_argument('--split', default='train', help='The set to get the interactions from')
    
    args = ap.parse_args()
    
    output_path = os.path.join(MODELS_PATH, 'llm/llama_cpp_python', f'ciao_{args.category.replace(" & ", "_")}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = run(args.category, args.signal, args.split)
    df.to_csv(os.path.join(output_path, f'{args.signal}_{args.split}_profiles.csv'), index=False)
    
    os.remove(os.path.join(os.path.dirname(__file__), 'profiles_temp.csv'))
    print('done')
