import os
import time

from tinygrad import Context, nn, Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad import dtypes
import matplotlib.pyplot as plt  # for making figures
from tinygrad import Device

# hyperparameters
block_size: int = 256  # Maximum sequence length for input and target
vocab_size: int = 65  # Size of the vocabulary (number of unique tokens)
n_layer: int = 6      # Number of transformer layers in the model
n_head: int = 6       # Number of attention heads in each transformer layer
n_embd: int = 384     # Dimensionality of the token embeddings and hidden states


class CausalSelfAttention:
    def __init__(self):
        assert n_embd % n_head == 0

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        k = k.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)
        q = q.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)
        v = v.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = q.scaled_dot_product_attention(
            k, v, attn_mask=None, dropout_p=0.2 if Tensor.training else 0, is_causal=True)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)

        # output projection
        y = self.c_proj(y).dropout(0.2)
        return y


class MLP:
    def __init__(self):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x).quick_gelu()
        x = self.c_proj(x)
        return x


class Block:
    def __init__(self):
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT:
    def __init__(self):
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = [Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
