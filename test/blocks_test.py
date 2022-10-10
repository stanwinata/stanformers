from src import blocks
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

def ReferenceAttention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class ReferenceMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(ReferenceMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = blocks.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = ReferenceAttention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class ReferencePositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(ReferencePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class ReferenceEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(ReferenceEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class ReferencePositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ReferencePositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

def testFFNetwork():
    vocab = 10
    d_model = 20
    d_ff = 40
    inputs = torch.FloatTensor(d_model)
    torch.manual_seed(0)
    ff_network  = blocks.FFNetwork(d_model, d_ff, dropout= 0.0)
    torch.manual_seed(0)
    ref_ff_net = ReferencePositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    y = ff_network.forward(inputs)
    ref_y = ref_ff_net.forward(inputs)
    rtol = 0.0000001
    max_delta = torch.max(torch.abs(y-ref_y))
    assert(rtol > max_delta)
    print("PASSED: Test FFNetwork")


def testEmbedding():
    vocab = 10
    d_model = 20
    inputs = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    torch.manual_seed(0)
    embed  = blocks.Embedding(vocab, d_model)
    torch.manual_seed(0)
    ref_embed = ReferenceEmbedding(vocab, d_model)
    y = embed.forward(inputs)
    ref_y = ref_embed.forward(inputs)
    rtol = 0.0000001
    max_delta = torch.max(torch.abs(y-ref_y))
    assert(rtol > max_delta)
    print("PASSED: Test Embedding")

def testPositionalEncoding():
    d_model = 20
    dropout = 0.0
    seq_len = 100
    batch_size = 5
    pe  = blocks.PositionalEncoding(d_model, dropout)
    ref_pe = ReferencePositionalEncoding(d_model, dropout)
    y = pe.forward(torch.zeros(batch_size, seq_len, d_model))
    ref_y = ref_pe.forward(torch.zeros(batch_size, seq_len, d_model))
    rtol = 0.0000001
    max_delta = torch.max(torch.abs(y-ref_y))
    assert(rtol > max_delta)
    print("PASSED: Test positional encoding")
    # Uncomment to visualize
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d"%p for p in [4,5,6,7]])
    # plt.show()

def testMultiHeadAttention():
    # TODO: Check with mask and dropout+seed.
    d_model = 24
    num_head = 4
    batch_size = 3
    sentence_length = 4
    Q = torch.randn([sentence_length, d_model])
    K = torch.randn([sentence_length, d_model])
    V = torch.randn([sentence_length, d_model])
    torch.manual_seed(0)
    ref_MHA = ReferenceMultiHeadedAttention(num_head, d_model, dropout=0.0)
    torch.manual_seed(0)
    # TODO: push up issue against annotated transformers repo.
    MHA = blocks.MultiHeadAttention(num_head, d_model, dropout=0.0)
    torch.manual_seed(0)
    torch_MHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_head)
    random_weight_float = 12.345
    with torch.no_grad():
        torch_MHA.in_proj_weight.fill_(random_weight_float)
        torch_MHA.out_proj.weight.fill_(random_weight_float)
        MHA.linear_Q.weight.fill_(random_weight_float)
        MHA.linear_K.weight.fill_(random_weight_float)
        MHA.linear_V.weight.fill_(random_weight_float)
        MHA.linear_O.weight.fill_(random_weight_float)

        torch_MHA.in_proj_bias.fill_(random_weight_float)
        torch_MHA.out_proj.bias.fill_(random_weight_float)
        MHA.linear_Q.bias.fill_(random_weight_float)
        MHA.linear_K.bias.fill_(random_weight_float)
        MHA.linear_V.bias.fill_(random_weight_float)
        MHA.linear_O.bias.fill_(random_weight_float)
    ref_y, _ = torch_MHA.forward(Q, K, V)
    y = MHA.forward(Q, K, V)
    y = y.squeeze()
    rtol = 0.0001
    max_delta = torch.max(torch.abs(y-ref_y))
    assert(rtol > max_delta)
    print("PASSED: Multi Head Attention")

if __name__ == "__main__":
    testPositionalEncoding()
    testEmbedding()
    testFFNetwork()
    testMultiHeadAttention()