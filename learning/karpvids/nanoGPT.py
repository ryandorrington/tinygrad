import os
import time

from tinygrad import Context, nn, Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad import dtypes
import numpy as np

import matplotlib.pyplot as plt  # for making figures
from tinygrad import Device
print(Device.DEFAULT)
# hyperparameters

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
num_heads = 6
head_size = n_embd // num_heads
# ------------

Tensor.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoder: take a string, output a list of integers
def encode(s): return [stoi[c] for c in s]
# decoder: take a list of integers, output a string
def decode(l): return ''.join([itos[i] for i in l])


# Train and test splits
data = Tensor(encode(text), dtype=dtypes.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def stack_data(data, ix, block_size, offset):
    ret = []
    for i in ix:
        data_slice = data[i.item()+offset:i.item()+block_size+offset].numpy()
        ret.append(data_slice)
    return Tensor(ret, dtype=dtypes.int64)


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = Tensor.randint((batch_size,), high=len(data) - block_size)
    x = stack_data(data, ix, block_size, offset=0)
    y = stack_data(data, ix, block_size, offset=1)
    if Device.DEFAULT == 'CUDA':
        x, y = x.to(Device.DEFAULT), y.to(Device.DEFAULT)
    return x, y


def estimate_loss():
    Tensor.no_grad = True
    Tensor.training = False
    out = {}

    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = Tensor(losses).mean()
    Tensor.training = True
    Tensor.no_grad = False
    return out

# super simple bigram model


class Head:
    def __init__(self):
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = Tensor.tril(Tensor.ones(block_size, block_size))
        self.tril.requires_grad = False

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax().dropout(0.2)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention:
    def __init__(self):
        self.h1 = Head()
        self.h2 = Head()
        self.h3 = Head()
        self.h4 = Head()
        self.h5 = Head()
        self.h6 = Head()
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        t = Tensor.cat(self.h1(x), self.h2(x), self.h3(
            x), self.h4(x), self.h5(x), self.h6(x), dim=-1)

        return self.proj(t).dropout(0.2)


class FeedForward:
    def __init__(self):
        self.net = [nn.Linear(n_embd, 4 * n_embd), Tensor.relu,
                    nn.Linear(4 * n_embd, n_embd)]

    def __call__(self, x):
        return x.sequential(self.net).dropout(0.2)


class Block:
    def __init__(self):
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel:

    def __init__(self):
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = [Block(), Block(), Block(), Block(), Block(), Block()]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(Tensor.arange(T))  # (T,C)
        # (B,T,C) + (T,C) -> (B,T,C) + (1,T,C) -> (B,T,C)
        x = tok_emb + pos_emb
        x = x.sequential(self.blocks)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = logits.cross_entropy(targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = logits.softmax().numpy()  # (B, C)
            # sample from the distribution

            idx_next = Tensor(
                [np.random.choice(len(probs[0]), p=probs[0])], dtype=dtypes.int64)
            # append sampled index to the running sequence
            idx = idx.cat(idx_next.unsqueeze(0), dim=1)  # (B, T+1)
        return idx


model = LanguageModel()


optimizer = AdamW(nn.state.get_parameters(model), lr=learning_rate)


@TinyJit
def train_step(xb, yb):
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


last_time = time.time()
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        current_time = time.time()
        elapsed_time = current_time - last_time
        last_time = current_time
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train'].item():.4f}, val loss {losses['val'].item():.4f}, time: {elapsed_time:.2f}s")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    loss = train_step(xb, yb)

# generate from the model
context = Tensor.zeros((1, 1), dtype=dtypes.long)
print(decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist()))
