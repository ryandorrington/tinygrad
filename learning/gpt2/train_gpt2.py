import os
from os import getenv
import time

from tinygrad import nn, Tensor, TinyJit, Device
from tinygrad.nn.optim import AdamW
from tinygrad import dtypes
from tinygrad.helpers import getenv

import numpy as np
import tiktoken

GPUS = [f'{Device.DEFAULT}:{i}' for i in range(getenv("GPUS", 2))]

# hyperparameters
block_size: int = 1024  # Maximum sequence length for input and target
vocab_size: int = 50257  # Size of the vocabulary (number of unique tokens)
n_layer: int = 12      # Number of transformer layers in the model
n_head: int = 12       # Number of attention heads in each transformer layer
n_embd: int = 768     # Dimensionality of the token embeddings and hidden states


class CausalSelfAttention:
    def __init__(self):
        assert n_embd % n_head == 0

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

        self.bias = Tensor.ones(block_size, block_size).reshape(1, 1, block_size, block_size)
        self.bias.requires_grad = False

    def __call__(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        k = k.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)
        q = q.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)
        v = v.reshape(B, T, n_head, C // n_head).transpose(1,
                                                           2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.softmax(axis=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.c_proj(y)
        return y


class MLP:
    def __init__(self):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x)
        return x


class Block:
    def __init__(self):
        self.attn = CausalSelfAttention()
        self.mlp = MLP()

    def __call__(self, x):
        x = x + self.attn(x.layernorm())
        x = x + self.mlp(x.layernorm())
        return x


class GPT:
    def __init__(self):
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = [Block() for _ in range(n_layer)]

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

    @classmethod
    def from_pretrained(cls):
        # create a from-scratch initialized minGPT model
        model = GPT()
        sd = nn.state.get_state_dict(model)
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        if not os.path.exists("hf_gpt2_weights.pth"):
            from get_hf_gpt2_weights import get_hf_gpt2_weights
            get_hf_gpt2_weights()
        
        sd_hf = nn.state.torch_load("hf_gpt2_weights.pth")

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            k_trimmed = k.replace('transformer.', '')  # Remove 'transformer.' prefix
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k_trimmed].shape
                Tensor.no_grad = True
                sd[k_trimmed].assign(Tensor(sd_hf[k].numpy()).transpose())
                Tensor.no_grad = False
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k_trimmed].shape
                Tensor.no_grad = True
                sd[k_trimmed].assign(Tensor(sd_hf[k].numpy()))
                Tensor.no_grad = False

        # Check if model's params are correctly set to the sd model's params
        for k in sd_keys:
            k_hf = 'transformer.' + k if k != 'lm_head.weight' else k
            if any(k.endswith(w) for w in transposed):
                assert np.allclose(sd[k].numpy(), sd_hf[k_hf].numpy().T, atol=1e-5), f"Param {k} not correctly set"
            else:
                assert np.allclose(sd[k].numpy(), sd_hf[k_hf].numpy(), atol=1e-5), f"Param {k} not correctly set"
        
        print("All parameters successfully loaded and verified.")
        return model
    
    def __call__(self, idx, targets=None):

        B, T = idx.size()
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        pos = Tensor.arange(0, T, dtype=dtypes.long) # shape (t)

        # forward the GPT model itself
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)
        x = x.layernorm()
        
        logits = self.lm_head(x)

        return logits

def generate_from_hf(): 
    num_return_sequences = 2
    max_length = 30

    model = GPT.from_pretrained()

    print(f"Using device: {Device.DEFAULT}")

    Tensor.training = False
    Tensor.no_grad = True

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = Tensor(tokens, dtype=dtypes.long)
    x = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    while x.size(1) < max_length:
        logits = model(x)
        logits = logits[:, -1, :]
        probs = logits.softmax(axis=-1)
        ix = probs.multinomial(num_samples=1)

        x = x.cat(ix, dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

    Tensor.training = True
    Tensor.no_grad = False



class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("tinyshakespeare.txt") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = Tensor(tokens).shard(GPUS, axis=0)

        print(f"Loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    


Tensor.training = True
train_loader = DataLoader(B=8, T=1024)
model = GPT()


optimizer = AdamW(nn.state.get_parameters(model), lr=3e-4)

def step():
    t0 = time.time()
    x, y = train_loader.next_batch()
    


    optimizer.zero_grad()
    logits = model(x)
    loss = logits.reshape(-1, logits.shape[-1]).cross_entropy(y.flatten()).backward()
    optimizer.step()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_second = (train_loader.B * train_loader.T) / (t1 - t0)
    return loss, dt, tokens_per_second

tiny_step = TinyJit(step)

print("TINYJIT ------------------------------")
for i in range(50):
    loss, dt, tokens_per_second = tiny_step()
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tokens_per_second: {tokens_per_second:.2f}")
