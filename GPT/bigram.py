import torch
import torch.nn as nn
from torch.nn import functional as F
import time


DATA_DIR = "/home/ubuntu/nihal/data/input.txt"
torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')
batch_size = 128
block_size = 256
max_iter = 500
eval_interval = 500
learning_rate = 3e-4
device = "cuda:1" if torch.cuda.is_available() else "cpu"
eval_iter = 200
n_embed = 384
# head_size = 16
n_head = 6
n_layer = 6
dropout = 0.2
epochs = 8

def read_data(data_dir):
    with open(data_dir, "r") as f:
        text = f.read()
    return text

text = read_data(DATA_DIR)
chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {c:idx for idx, c in enumerate(chars)}
itos = {idx:c for idx, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def split_data(data):
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    return train_data, val_data

train_data, val_data = split_data(data)

def get_batch(split):
    data = train_data if split == "train" else val_data
    sample = torch.randint(high=len(data)-block_size, size=(batch_size,))
    x = torch.stack([data[s:s+block_size] for s in sample])
    y = torch.stack([data[s+1:s+block_size+1] for s in sample])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''Single headed self attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention in parallel'''
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    '''Simple linear layer followed by non linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''Transformer Block'''

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.block = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        # B = batch_size, T = seq_len, C = output_classes
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_embed
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


lm = BigramLanguageModel()
lm = lm.to(device)
lm = torch.compile(lm)
optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)

for epoch in range(epochs):

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step in range(max_iter):

        if step % eval_interval == 0:
            losses = estimate_loss(lm)
            print(f"Train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        
        logits, loss = lm(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Time elapsed for epoch {epoch+1}: {elapsed_time:.2f} seconds")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(lm.generate(context, max_new_tokens=5000)[0].tolist()))