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
    query, key, value : tensor<h x sent_len x d_K>
    Query * Key_transpose -> scaled_score : tensor <h x sent_len x sent_len>
    attention_val = scaled_score * value : (h x sent_len x sent_len), (h x sent_len x sent_len) -> (h x sent_len x d_k)
    where h = number of heads
    where sent_len = sentence length
    where d_k = dimensionality in multiple head. (typically model_dimensionality / num_head)

    Matrix with shape sent_len x sent_len represents
    how each word affect others in the sentence.

    We use multiple head on top of this to
    learn different types of relationships between words.
    """
    num_head, batch, d_k = key.shape
    scaled_scores = torch.bmm(query, key.transpose(-2, -1))/math.sqrt(d_k)
    # Ensure value of illegal connection not be used by setting weight to -Inf.
    if mask is not None:
        scaled_scores.masked_fill(mask == 0, 1e-9)
    scaled_weight = scaled_scores.softmax(dim=-1)
    if dropout is not None:
        scaled_weight = dropout(scaled_weight)
    attention_val = torch.bmm(scaled_weight, value)
    # Concatenating multiple heads into one.
    attention_val = attention_val.transpose(0,1).reshape(batch, num_head * d_k)
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

        # Split up the linear projections from sentence_len x d_model
        # => sentence_len x (h x d_k).
        # => h x sentence_len x d_k. for easy computation
        multi_head_query = proj_query.view(nbatches, self.h, self.d_k).transpose(0,1)
        multi_head_key = proj_key.view(nbatches, self.h, self.d_k).transpose(0,1)
        multi_head_value = proj_value.view(nbatches, self.h, self.d_k).transpose(0,1)

        # Apply attention to h x
        x = attention(multi_head_query, multi_head_key, multi_head_value, mask, self.dropout)
        # Concat attention
        x = (
            x.contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        # Apply project to output.
        return self.linear_O(x)
