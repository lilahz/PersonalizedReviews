import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class SentenceEmbedding:
    def __init__(self):
        # Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, review_sentences):
        # Sentences we want sentence embeddings for
        sentences = ['This framework generates embeddings for each input sentence',
                     'Sentences are passed as a list of string.',
                     'The quick brown fox jumps over the lazy dog.']

        # Tokenize sentences
        encoded_input = self.tokenizer(review_sentences,
                                       padding=True,
                                       truncation=True,
                                       max_length=128,
                                       return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])


class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim=100, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, in_dim]
        Returns:
            outputs, weights: [B, seq_len, out_dim], [B, seq_len]
        """
        # weights = self.proj(context) @ self.v
        weights = self.proj_v(self.proj(context)).squeeze(-1)
        weights = torch.softmax(weights, dim=-1)  # [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1), weights  # [B, 1, seq_len], [B, seq_len, dim]


class ReviewEncoder(nn.Module):
    def __init__(self, hparams, weight=None):
        super(ReviewEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300)
        else:
            self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=0)
        self.mha = nn.MultiheadAttention(hparams['embed_size'], num_heads=hparams['nhead'],
                                         dropout=self.hparams['dropout'])
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, x):
        x = F.dropout(self.embedding(x), 0.2)
        x = x.permute(1, 0, 2)
        output, _ = self.mha(x, x, x)
        output = F.dropout(output.permute(1, 0, 2), p=self.hparams['dropout'])
        output = self.proj(output)
        output, _ = self.additive_attn(output)
        return output
    
    
class LLMReviewEncoder(torch.nn.Module):
    def __init__(self, hparams, llm_model):
        super(LLMReviewEncoder, self).__init__()
        self.hparams = hparams
        self.llm_model = llm_model
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hparams['llm_review_size'],
                                                         num_heads=self.hparams['llm_nhead'],  # 16, 32
                                                         dropout=self.hparams['dropout'])
        self.additive_attention = AdditiveAttention(self.hparams['llm_review_size'], self.hparams['v_size'])

    def forward(self, text_ids, text_attmask):
        # batch_size, num_words_text
        # batch_size, num_words = text.shape
        # num_words = num_words // 3
        # text_ids = torch.narrow(text, 1, 0, num_words)
        # text_attmask = torch.narrow(text, 1, num_words * 2, num_words)
        word_emb = self.llm_model(text_ids, text_attmask)[0]
        text_vector = F.dropout(word_emb, p=self.hparams['dropout'])
        # batch_size, num_words_text, word_embedding_dim
        multihead_text_vector, _ = self.multihead_attention(text_vector, text_vector, text_vector)
        multihead_text_vector = F.dropout(multihead_text_vector, p=self.hparams['dropout'])
        # batch_size, word_embedding_dim
        text_vector, _ = self.additive_attention(multihead_text_vector)
        return text_vector


class NRMS(nn.Module):
    def __init__(self, hparams, weight=None, llm_model=None):
        super(NRMS, self).__init__()
        self.hparams = hparams
        if self.hparams['embedding_type'] == 'llm':
            self.review_encoder = LLMReviewEncoder(hparams, llm_model)
            self.mha = nn.MultiheadAttention(hparams['llm_review_size'], hparams['llm_nhead'], dropout=self.hparams['dropout'])
            self.additive_attn = AdditiveAttention(hparams['llm_review_size'], hparams['v_size'])
        else:
            self.review_encoder = ReviewEncoder(hparams, weight=weight)
            self.mha = nn.MultiheadAttention(hparams['embed_size'], hparams['nhead'], dropout=self.hparams['dropout'])
            self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size'])
        self.criterion = nn.CrossEntropyLoss()
        self.eval_criterion = nn.BCEWithLogitsLoss()

    def forward(self, history, candidates, labels=None, is_eval=None, history_attn=None, candidates_attn=None):
        """forward

        Args:
            history (tensor): [num_user, num_click_docs, seq_len] (written/liked reviews)
            candidates (tensor): [num_user, num_candidate_docs, seq_len] (product reviews)
        """
        num_history = history.shape[1]
        num_cand = candidates.shape[1]
        num_user = history.shape[0]
        if self.hparams['embedding_type'] == 'llm':
            ids_length = candidates.size(2)
            cand_embed = self.review_encoder(candidates.view(-1, ids_length), candidates_attn.view(-1, ids_length))
            
            if is_eval:
                cand_embed = cand_embed.view(-1, candidates.size(1), self.hparams['llm_review_size'])
            else:
                cand_embed = cand_embed.view(-1, 1 + self.hparams['npratio'], self.hparams['llm_review_size'])
            
            history_embed = self.review_encoder(history.view(-1, ids_length), history_attn.view(-1, ids_length))
            history_output = history_embed.view(-1, self.hparams['his_size'], self.hparams['llm_review_size'])
            
        else:
            seq_len = history.shape[2]
            history = history.reshape(-1, seq_len)
            candidates = candidates.reshape(-1, seq_len)
            history_embed = self.review_encoder(history)
            cand_embed = self.review_encoder(candidates)
            history_embed = history_embed.reshape(num_user, num_history, -1)
            cand_embed = cand_embed.reshape(num_user, num_cand, -1)
            history_embed = history_embed.permute(1, 0, 2)
            history_output, _ = self.mha(history_embed, history_embed, history_embed)
            # history_output = F.dropout(click_output.permute(1, 0, 2), 0.2)
            history_output = F.dropout(history_output.permute(1, 0, 2), self.hparams['dropout'])
            user_repr = self.proj(history_output)
            
        user_repr, _ = self.additive_attn(history_output)

        logits = torch.bmm(user_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1)  # [B, 1, hid], [B, 10, hid]
        
        if labels is not None:
            if is_eval:
                loss = self.eval_criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits
            # return torch.sigmoid(logits)


class NRMSCombined(nn.Module):
    def __init__(self, hparams, weight=None):
        super(NRMSCombined, self).__init__()
        self.hparams = hparams
        self.review_encoder = ReviewEncoder(hparams, weight=weight)
        # self.mha = nn.MultiheadAttention(hparams['embed_size'], hparams['nhead'], dropout=0.1)
        self.mha = nn.MultiheadAttention(hparams['embed_size'], hparams['nhead'], dropout=self.hparams['dropout'])
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])
        self.criterion = nn.CrossEntropyLoss()
        self.eval_criterion = nn.BCEWithLogitsLoss()

    def forward(self, write_reviews, like_reviews, candidates, labels=None, is_eval=False):
        """forward

        Args:
            write_reviews (tensor): [num_user, num_click_docs, seq_len] (written reviews by the user)
            like_reviews (tensor): [num_user, num_click_docs, seq_len] (liked reviews by the user)
            candidates (tensor): [num_user, num_candidate_docs, seq_len] (product reviews to be ranked)
        """
        num_w_revs = write_reviews.shape[1]
        num_l_revs = like_reviews.shape[1]
        num_cand_revs = candidates.shape[1]
        num_user = write_reviews.shape[0]
        seq_len = write_reviews.shape[2]

        write_reviews = write_reviews.reshape(-1, seq_len)
        write_reviews_embed = self.review_encoder(write_reviews)
        write_reviews_embed = write_reviews_embed.reshape(num_user, num_w_revs, -1)

        like_reviews = like_reviews.reshape(-1, seq_len)
        like_reviews_embed = self.review_encoder(like_reviews)
        like_reviews_embed = like_reviews_embed.reshape(num_user, num_l_revs, -1)

        candidates = candidates.reshape(-1, seq_len)
        cand_embed = self.review_encoder(candidates)
        cand_embed = cand_embed.reshape(num_user, num_cand_revs, -1)

        write_reviews_embed = write_reviews_embed.permute(1, 0, 2)
        write_reviews_output, _ = self.mha(write_reviews_embed, write_reviews_embed, write_reviews_embed)
        write_reviews_output = F.dropout(write_reviews_output.permute(1, 0, 2), self.hparams['dropout'])

        like_reviews_embed = like_reviews_embed.permute(1, 0, 2)
        like_reviews_output, _ = self.mha(like_reviews_embed, like_reviews_embed, like_reviews_embed)
        like_reviews_output = F.dropout(like_reviews_output.permute(1, 0, 2), self.hparams['dropout'])

        # batch, write history dimension + like history dimension, embedded output
        combined_output = torch.cat((write_reviews_output, like_reviews_output), 1)  # 1- 2nd dim, 2-on 3rd dim
        # print(combined_output.shape)
        user_repr, _ = self.additive_attn(combined_output)

        logits = torch.bmm(user_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1)  # [B, 1, hid], [B, 10, hid]
        if labels is not None:
            if is_eval:
                loss = self.eval_criterion(logits, labels)
            else:
                loss = self.criterion(logits, labels)
            return loss, logits
        else:
            return logits
