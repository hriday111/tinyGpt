## @file sdp_attention.py
#  @brief Character-level Transformer with scaled dot-product self-attention.
#
#  Extends the bigram baseline with multi-head causal attention, residual
#  blocks, LayerNorm, and a feed-forward MLP — following Karpathy's nanoGPT
#  teaching progression.
#
#  Expects `input.txt` (Shakespeare corpus) in the working directory.
#  Target hyperparameters can exceed ~4 GB local VRAM; cloud T4 training is
#  the intended path for the full config.

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
train_data_amount_coeff = 0.9
device = "cpu"
if torch.cuda.is_available():
    print("Using CUDA")
    device = "cuda"
eval_iters = 200
no_embbeddings = 384
no_head = 6
no_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}  # look into using tiktoken
itos = {i: ch for i, ch in enumerate(chars)}


## @brief Encode a string into a list of character token ids.
#  @param s Plain-text string.
#  @return List of integer token ids.
def encode(s):
    return [stoi[c] for c in s]


## @brief Decode a list of token ids back into a string.
#  @param l Sequence of integer token ids.
#  @return Decoded plain-text string.
def decode(l):
    return "".join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_data_amount_coeff * len(data))
train_data = data[:n]
val_data = data[n:]


## @brief Sample a random batch of context/target pairs.
#  @param split Either `"train"` or `"val"`.
#  @return Tuple `(x, y)` of shape `(batch_size, block_size)` on `device`.
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


## @brief Estimate mean train and validation loss without gradients.
#  @return Dict with keys `"train"` and `"val"` (scalar tensors).
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


## @brief One causal self-attention head (scaled dot-product).
#
#  Computes key/query/value projections, applies a lower-triangular mask so
#  tokens only attend to the past, then softmax-weights the values.
class Head(nn.Module):
    ## @brief Create K/Q/V linears and the causal mask buffer.
    #  @param head_size Per-head channel width (`n_embd // n_head`).
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(no_embbeddings, head_size, bias=False)
        self.query = nn.Linear(no_embbeddings, head_size, bias=False)
        self.value = nn.Linear(no_embbeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    ## @brief Apply masked self-attention to a token sequence.
    #  @param x Embeddings of shape `(B, T, C)`.
    #  @return Head output of shape `(B, T, head_size)`.
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # affinities (attention score) compute
        wei = (
            q @ k.transpose(-2, -1) * C ** (-0.5)
        )  # (B,T,16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        out = wei @ self.value(x)
        return out


xb, yb = get_batch("train")

"""for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        """


## @brief Parallel self-attention heads concatenated and projected back.
class MultiHeadAttention(nn.Module):
    ## @brief Build `num_heads` attention heads plus output projection.
    #  @param num_heads Number of parallel heads.
    #  @param head_size Channels per head.
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(no_embbeddings, no_embbeddings)
        self.dropout = nn.Dropout(dropout)

    ## @brief Run all heads and project the concatenated result.
    #  @param x Embeddings of shape `(B, T, C)`.
    #  @return Tensor of shape `(B, T, C)`.
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


## @brief Position-wise MLP applied after attention (expand → ReLU → project).
class FeedForward(nn.Module):
    ## @brief Build the 4× widening feed-forward network.
    #  @param no_embbeddings Model embedding dimension.
    def __init__(self, no_embbeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(no_embbeddings, 4 * no_embbeddings),
            nn.ReLU(),
            nn.Linear(4 * no_embbeddings, no_embbeddings),
            nn.Dropout(dropout),
        )

    ## @brief Apply the feed-forward network.
    #  @param x Embeddings of shape `(B, T, C)`.
    #  @return Tensor of shape `(B, T, C)`.
    def forward(self, x):
        return self.net(x)


## @brief Transformer block: LayerNorm → attention → residual → LayerNorm → FFN → residual.
class Block(nn.Module):
    ## @brief Wire multi-head attention and feed-forward with pre-norm residuals.
    #  @param no_embbeddings Model embedding dimension.
    #  @param n_head Number of attention heads.
    def __init__(self, no_embbeddings, n_head):
        super().__init__()
        head_size = no_embbeddings // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(no_embbeddings)

        self.ln1 = nn.LayerNorm(no_embbeddings)
        self.ln2 = nn.LayerNorm(no_embbeddings)

    ## @brief One Transformer block forward pass.
    #  @param x Embeddings of shape `(B, T, C)`.
    #  @return Tensor of shape `(B, T, C)`.
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


## @brief Full character-level language model (embeddings + Transformer stack + LM head).
class BigramLanguageModel(nn.Module):
    ## @brief Create token/position embeddings, `n_layer` blocks, and the LM head.
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, no_embbeddings)
        self.position_embedding_table = nn.Embedding(block_size, no_embbeddings)
        self.blocks = nn.Sequential(
            *[Block(no_embbeddings, n_head=no_head) for _ in range(no_layer)]
        )
        self.ln_f = nn.LayerNorm(no_embbeddings)
        self.lm_head = nn.Linear(no_embbeddings, vocab_size)

    ## @brief Forward pass through the Transformer LM.
    #  @param idx Token indices of shape `(B, T)`.
    #  @param targets Optional target indices of shape `(B, T)`.
    #  @return `(logits, loss)` where `loss` is `None` if `targets` is omitted.
    def forward(self, idx, targets=None):
        # targets  are B*T
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # (B T C)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    ## @brief Autoregressively sample new tokens (context cropped to `block_size`).
    #  @param idx Starting context indices of shape `(B, T)`.
    #  @param max_new_tokens How many tokens to append.
    #  @return Extended index tensor of shape `(B, T + max_new_tokens)`.
    def generate(self, idx, max_new_tokens):
        # idx is (B,T ) aree of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # gets the predictions
            logits = logits[
                :, -1, :
            ]  # focus on the last time step and it changes to (B,)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)


for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, validate loss {losses['val']:.4f}"
        )
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
