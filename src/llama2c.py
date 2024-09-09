"""
With some modifcations from https://github.com/karpathy/llama2.c
"""

import math
import typing as tp

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def precompute_freqs_cis(dim:int, end:int, theta:float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis:Tensor, x:Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq:Tensor,
    xk:Tensor,
    freqs_cos:Tensor,
    freqs_sin:Tensor
) -> tuple[Tensor, Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.n_kv_heads = getattr(config, "n_kv_heads", config.num_heads)
        assert config.num_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.num_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.wq = nn.Linear(config.d_model, config.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.num_heads * self.head_dim, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.resid_dropout = nn.Dropout(config.dropout_rate)
        self.dropout = config.dropout_rate

    def forward(
        self,
        x:Tensor,
        freqs_cos:Tensor,
        freqs_sin:Tensor,
        mask:Tensor
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask[:, :, :seqlen, :seqlen],
            dropout_p=self.dropout if self.training else 0.0
        )

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim:tp.Optional[int], dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class TransformerBlock(nn.Module):
    def __init__(self, layer_id:int, config:tp.Any):
        super().__init__()
        self.n_heads = config.num_heads
        self.dim = config.d_model
        self.head_dim = config.d_model // config.num_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.d_model,
            hidden_dim=None,
            dropout=config.dropout_rate,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.RMSNorm(config.d_model, eps=1e-5)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=1e-5)

    def forward(self, x, freqs_cos, freqs_sin, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    def __init__(self, config:tp.Any):
        super().__init__()
        self.n_layers = config.num_layers

        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.num_layers):
            self.layers.append(TransformerBlock(layer_id, config))
        self.norm = nn.RMSNorm(config.d_model, eps=1e-5)

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(config.d_model // config.num_heads, config.maxlen)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, h:Tensor, mask:Tensor) -> Tensor:
        seqlen = h.size(1)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, mask)
        h = self.norm(h)
        return h
