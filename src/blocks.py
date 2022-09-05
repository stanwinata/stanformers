import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
    Encoding position of inputs/sequence using sine and cosine.
    Using different frequencies to help embed positioning.

    Formula:
        PE[pos,2i] = sin(pos/10000^(2i/dmodel))
        PE[pos,2i+1] = cos(pos/10000^(2i/dmodel))
    Args:
        pos: every position in sequence (All elements in [0,SEQ_LEN])
        2i: dimensionality ind from [0,d_model] jumping every 2 steps.
        2i+1: dimensionality ind from [1,d_model] jumping every 2 steps.
        d_model: embedding size/dimensionality.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Arguments:
            d_model(int~[1,+Inf]): Embedding size and dimensionality of input and output of attention layers.
            dropout(float~[0,1)): Percentage/rate of neurons being dropped for regularization.
            max_len(int~[1,+Inf]): Maximum length of input sequence.
        Returns:
            PE(tensor<seq_len x d_model x float>): Positional encodings of every position in sequence to be added to the embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        positional_encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        """
        Using log space:
        Theory: a*log(x) <=> log(x^a) AND exp(log(x)) <=> x.
        => 10000^(2i/dmodel) = exp(log(10000) * 2i / d_model).
        => 1/(10000^(2i/dmodel)) = 10000^(-2i/dmodel) = exp(log(10000) * -2i / d_model).
        => Move scalars to the left for more efficient compute: exp(-log(10000)/d_model * 2i).
        """
        div_term = torch.exp(-math.log(10000.0) / d_model * torch.arange(0,d_model,2))

        # Set indices to mimic 2i (0,2,4,...)
        positional_encoding[:,0::2] = torch.sin(pos * div_term)
        # Set indices to mimic 2i+1 (1,3,5,...)
        positional_encoding[:,1::2] = torch.cos(pos * div_term)
        # Setup batch dimension and register as torch buffer.
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("pe", positional_encoding)

    def forward(self, x):
        """
        To this end, we add "positional encodings" to the input embeddings at the
        bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
        as the embeddings, so that the two can be summed.
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Embedding(nn.Module):
    """
    A NxD Learnable Lookup table/embedding, where N is size of vocabulary, and D is dimension.

    Args:
        vocab(int ~[1,+Inf]): size of vocabulary
        d_model(int ~[1,+Inf]): embedding size/dimensionality.
    """
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.lut_table = nn.Embedding(num_embeddings=vocab, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Given indices of words, return the corresponding learned embeddings vector.

        Args:
            x(tensor<ix32>): tensor of word indices to query.
        """
        return self.lut_table(x) * math.sqrt(self.d_model)

class FFNetwork (nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes and defines the FFNetwork.
        FFN(x) = max(0, xW1 + b1)W2 + b2 (2)

        Arguments:
            d_model(int~[1,+Inf]): Embedding size and dimensionality of input and output of attention layers.
                                   Also, input and output dimension for the FFNetwork in Transformers.
            d_ff(int~[1,+Inf]): Dimensionality of hidden layer/inner-layer
            dropout(float~[0,1)): Percentage/rate of neurons being dropped for regularization.
        """
        super(FFNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

def attention(query, key, value, mask=None, dropout=None):
    """
    query, key, value : tensor<batch x head x d_k>
    Since each head is independent, we can let batch-prime = batch x head.
    query, key, value : tensor<batch-prime x d_k>

    then scaled_score = query * key_transpose : (batch-prime x 1 x d_k), (batch-prime x d_k x 1)
    then attention_val = scaled_score * value : (batch-prime x 1), (batch-prime x d_k) -> (batch_prime x d_k)
    """
    d_k = key.shape[-1]
    scaled_scores = torch.einsum('ijk,ikj->ij', query, key.transpose(-2, -1))/math.sqrt(d_k)
    # Ensure value of illegal connection not be used by setting weight to -Inf.
    if mask is not None:
        scaled_scores.masked_fill(mask == 0, 1e-9)
    scaled_weight = scaled_scores.softmax(dim=-1)
    if dropout is not None:
        scaled_weight = dropout(scaled_weight)
    attention_val = torch.einsum('ij,ijk->ijk',scaled_weight, value)
    return attention_val

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        """
        num_heads = number of heads/parallel attention layers.
        d_model = model dimensionality.
        """
        super(MultiHeadAttention, self).__init__()
        # Want to split model dimensionality by num heads
        # s.t after concat, dim of multi-head attn ~= d_model.
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.linear_K, self.linear_Q, self.linear_V, self.linear_O = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Apply linear projections. In this implementation,
        # we combine the h x d_k projections into a single linear layer
        # and split it out lkater for better performance.
        proj_query = self.linear_K(query)
        proj_key = self.linear_Q(key)
        proj_value = self.linear_V(value)
        nbatches = query.shape[0]

        # Split up the linear projections from d_model => h x d_k.
        multi_head_query = proj_query.view(nbatches, self.h, self.d_k)
        multi_head_key = proj_key.view(nbatches, self.h, self.d_k)
        multi_head_value = proj_value.view(nbatches, self.h, self.d_k)

        # Apply attention to h x
        x = attention(multi_head_query, multi_head_key, multi_head_value, mask, self.dropout)
        # Concat attention
        x = (
            x.contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        # Apply project to output.
        return self.linear_O(x)
