import os
import time

from tinygrad import Context, nn, Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad import dtypes
from tinygrad import Device

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
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def __call__(self, x):
        x = self.c_fc(x).quick_gelu()
        x = self.c_proj(x)
        return x


class Block:
    def __init__(self):
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT:
    def __init__(self):
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = [Block() for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

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

        # print("sd_keys_hf: ", sd_keys_hf)
        print("sd_keys: ", sd_keys)
        for k in sd_keys_hf:
            k_trimmed = k.replace('transformer.', '')  # Remove 'transformer.' prefix
            if any(k_trimmed.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k_trimmed].shape
                Tensor.no_grad = True
                sd[k_trimmed] = sd_hf[k].T
                Tensor.no_grad = False
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k_trimmed].shape
                Tensor.no_grad = True
                sd[k_trimmed] = sd_hf[k]
                Tensor.no_grad = False

        return model

# -----------------------------------------------------------------------------

model = GPT.from_pretrained()
print("ran")