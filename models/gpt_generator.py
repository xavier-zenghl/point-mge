# models/gpt_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj_drop(self.proj(out))


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTGenerator(nn.Module):
    def __init__(self, codebook_size=8192, embed_dim=384, depth=12, num_heads=6, seq_len=64, num_classes=55, dropout=0.1):
        super().__init__()
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.tok_embed = nn.Embedding(codebook_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.class_embed = nn.Embedding(num_classes, embed_dim) if num_classes > 0 else None
        self.drop = nn.Dropout(dropout)
        total_seq = seq_len + (1 if num_classes > 0 else 0)
        self.blocks = nn.ModuleList([GPTBlock(embed_dim, num_heads, total_seq, 4.0, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, codebook_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, indices, class_label=None):
        B, L = indices.shape
        tok = self.tok_embed(indices) + self.pos_embed[:, :L]
        if self.class_embed is not None and class_label is not None:
            cls_tok = self.class_embed(class_label).unsqueeze(1)
            x = torch.cat([cls_tok, tok], dim=1)
        else:
            x = tok
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.class_embed is not None and class_label is not None:
            x = x[:, 1:]
        return self.head(x)

    @torch.no_grad()
    def generate(self, class_label=None, temperature=1.0, top_k=0, top_p=1.0):
        B = class_label.shape[0] if class_label is not None else 1
        device = next(self.parameters()).device
        generated = torch.zeros(B, 0, dtype=torch.long, device=device)
        for i in range(self.seq_len):
            input_ids = torch.zeros(B, 1, dtype=torch.long, device=device) if generated.shape[1] == 0 else generated
            logits = self.forward(input_ids, class_label)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                top_k_vals, _ = next_logits.topk(top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = next_logits.sort(descending=True)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
