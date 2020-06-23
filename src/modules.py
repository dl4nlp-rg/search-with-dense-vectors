import torch
import torch.nn as nn
from transformers import BertModel

class CosineSimilarityLoss(nn.Module):
    """Loss based on cosine similarity for embeddings."""
    def __init__(self, k=1, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps
        self.sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.k = k

    def forward(self, q, d_pos, d_neg):
        sim_pos = self.sim(q, d_pos)
        sim_neg = self.sim(q, d_neg)
        loss = -torch.log(self.eps +
                torch.exp(sim_pos)/(torch.exp(sim_pos) + self.k*torch.exp(sim_neg))
                )
        return  loss, sim_pos, sim_neg

class Encoder(nn.Module):
    """BERT based encoder with reduced dim at the end."""
    def __init__(self, dim, pretrained_model='bert-base-uncased'):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(self.bert.config.hidden_size, dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, h = self.bert(input_ids=input_ids.squeeze(-2),
                         attention_mask=attention_mask.squeeze(-2),
                         token_type_ids=token_type_ids.squeeze(-2))
        logits = self.linear(self.activation(h))
        return logits
