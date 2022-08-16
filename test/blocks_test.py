from src import blocks
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

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
    batch_size = 1
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

if __name__ == "__main__":
    testPositionalEncoding()
    testEmbedding()
    testFFNetwork()