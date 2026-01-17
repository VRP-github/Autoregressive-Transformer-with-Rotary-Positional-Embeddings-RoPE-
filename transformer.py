import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset
import os

batch_size = 32
block_size = 256
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using device: {device}')
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

print("Loading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
text = "\n\n".join(ds['text'][:20000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.ByteLevel()

if not os.path.exists("tokenizer.json"):
    print("Training Wiki-based BPE tokenizer...")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]","[SEP]","[MASK]"]
    )
    with open("temp_wiki.txt", "w", encoding="utf-8") as f: f.write(text)
    tokenizer.train(["temp_wiki.txt"], trainer)
    tokenizer.save("tokenizer.json")
    os.remove("temp_wiki.txt")
else:
    tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
torch.manual_seed(1337)

vocab_size = tokenizer.get_vocab_size()

encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

encoded_ids = encode(text)
data = torch.tensor(encoded_ids, dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def perplexity_loss(loss):
    return torch.exp(loss).item()


def rotary_frequencies(dim, sequence_length, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(sequence_length, device=device).type_as(inv_freq)
    freqs = torch.einsum("i, j-> ij", t, inv_freq)
    emb = torch.cat((freqs, freqs),dim=-1)
    return emb.cos(), emb.sin()

def rotary_emb(x, cos, sin):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated_x * sin)


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        q = rotary_emb(q, cos[:T, :], sin[:T, :])
        k = rotary_emb(k, cos[:T, :], sin[:T, :])
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        cos, sin = rotary_frequencies(n_embd // n_head, T, device)
        for block in self.blocks:
            x = block(x, cos, sin)

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

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)
best_val_loss = float('inf')
start_iter = 0
CHECKPOINT_PATH = 'arxiv_checkpoint.pth'


if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iteration']

for iter in range(start_iter, max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_ppl = perplexity_loss(losses['train'])
        val_ppl = perplexity_loss(losses['val'])

        curr_lr = optimizer.param_groups[0]['lr']
        print(f"step {iter}: train loss {losses['train']:.4f} (PPL: {train_ppl:.2f}), val loss {losses['val']:.4f} (PPL: {val_ppl:.2f}) | LR: {curr_lr:.6f}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iteration': iter,
                'val_loss': best_val_loss
            }, 'model.pth')
            print(f"   --> New best validation loss. Saved to model.pth")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    scheduler.step()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iteration': iter,
}, CHECKPOINT_PATH)
print(f"Model checkpoint saved to: {CHECKPOINT_PATH}")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))