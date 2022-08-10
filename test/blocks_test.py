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