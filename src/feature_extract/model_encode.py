import torch
from torch import nn
from transformers import AutoModel, WavLMModel, BertModel


class PhoBERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=1024, projection_dim=512):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-large")
        self.project = nn.Sequential( 
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        projected = self.project(pooled_output)
        return pooled_output, projected


class BERTEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=1024, projection_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled = output.pooler_output
        return pooled, self.project(pooled)

class WavLMEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=1024, projection_dim=512):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.projection = nn.Sequential(  
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, input_values):
        outputs = self.wavlm(input_values=input_values)
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states.mean(dim=1)  
        projected = self.projection(pooled_output)
        return pooled_output, projected
