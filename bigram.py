## @file bigram.py
#  @brief Character-level bigram language model baseline (Karpathy nanoGPT style).
#
#  Trains a next-token model that looks only at the current character (no long
#  context). Useful as a loss/quality baseline before the self-attention model
#  in sdp_attention.py.
#
#  Expects `input.txt` (Shakespeare corpus) in the working directory.

import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
train_data_amount_coeff = 0.9
device = "cpu"
if torch.cuda.is_available():
    print("Using CUDA")
    device = "cuda"
eval_iters = 200


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


xb, yb = get_batch("train")

"""for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        """


## @brief Bigram next-token model via a vocab-sized embedding lookup.
#
#  Each token id maps directly to a distribution over the next token
#  (embedding dim == vocab size). No attention or positional encoding.
class BigramLanguageModel(nn.Module):
    ## @brief Build the token embedding table.
    #  @param vocab_size Number of distinct characters in the corpus.
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    ## @brief Forward pass: logits and optional cross-entropy loss.
    #  @param idx Token indices of shape `(B, T)`.
    #  @param targets Optional target indices of shape `(B, T)`.
    #  @return `(logits, loss)` where `loss` is `None` if `targets` is omitted.
    def forward(self, idx, targets=None):
        # targets  are B*T
        logits = self.token_embedding_table(idx)  # (B T C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    ## @brief Autoregressively sample new tokens from the model.
    #  @param idx Starting context indices of shape `(B, T)`.
    #  @param max_new_tokens How many tokens to append.
    #  @return Extended index tensor of shape `(B, T + max_new_tokens)`.
    def generate(self, idx, max_new_tokens):
        # idx is (B,T ) aree of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # gets the predictions
            logits = logits[
                :, -1, :
            ]  # focus on the last time step and it changes to (B,)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
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
