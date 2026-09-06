import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int

from cs336_basics.nn_utils import *


def get_tensor(in_features: int, out_features: int, device=None, dtype=None):
    # 按照剪裁正态分布初始化 weight
    weight = nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
    std = math.sqrt(2 / (in_features + out_features))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return weight
    
    
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = get_tensor(out_features, in_features, device, dtype) # (out_feature, in_feature)
        
    def forward(self, x: torch.Tensor):
        return torch.einsum("...i,oi->...o", x, self.weight) # (..., in_feature) @ (out_feature, in_feature) -> (..., out_feature)
    
    
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.weight = get_tensor(vocab_size, d_model, device, dtype)
        
    def forward(self, token_ids: Tensor):
        return self.weight[token_ids]

    
class Silu(nn.Module):
    # 激活函数；用在 SwiGLU 层
    def __init__(self):
        super().__init__()
    
    def forward(self, in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return in_features * torch.sigmoid(in_features)
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights: Float[Tensor, "d_model"]=None, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        if weights is None:
            self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        else:
            self.weight = nn.Parameter(weights.to(device=device, dtype=dtype))
        
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = in_features.dtype
        x = in_features.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # [batch_size, seq_len, d_model=1]
        return (x / rms * self.weight.to(torch.float32)).to(in_dtype) # [batch_size, seq_len, d_model]
        
        
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        x, in_dtype = in_features.to(torch.float32), in_features.dtype # [batch_size, seq_len, d_model]
        mu = x.mean(dim=-1, keepdim=True) # [batch_size, seq_len, 1]
        sigma = (x - mu).pow(2).mean(dim=-1, keepdim=True) # [batch_size, seq_len, 1]
        to_ret = ((x - mu) / torch.sqrt(sigma + self.eps)) * self.weight.to(torch.float32) + self.bias.to(torch.float32)
        return to_ret.to(in_dtype)
        
        
class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        w1_weight: Float[Tensor, " d_ff d_model"] = None, 
        w2_weight: Float[Tensor, " d_model d_ff"] = None, 
        w3_weight: Float[Tensor, " d_ff d_model"] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.weight1 = Linear(d_model, d_ff, device, dtype)
        self.weight2 = Linear(d_ff, d_model, device, dtype)
        self.weight3 = Linear(d_model, d_ff, device, dtype)
        self.silu = Silu()
        
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.weight2(self.silu(self.weight1(in_features)) * self.weight3(in_features))
    

class RoPE(nn.Module):
    def __init__(
        self, 
        d_k: int,
        theta: float,
        max_seq_len: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) # [d_k / 2]
        inv_freq = theta ** (-pair_indices / d_k) # [d_k / 2] 计算出每对 token 的旋转频率
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) # [max_seq_len]
        angles = torch.outer(positions, inv_freq) # [max_seq_len] @ [d_k/2] -> [max_seq_len, d_k / 2]
        self.cos = torch.cos(angles) # [max_seq_len, d_k/2]
        self.sin = torch.sin(angles) # [max_seq_len, dk/2]
        
    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        为什么要 token_positions，而不直接 arange(x.seq_len)？
        KV-Cache：
            输入的 token_position 就是输入 tensor 的 index：[0, 1, ..., t-1]
            这个的目的是在使用 kv cache 进行推理计算的时候，不会每次都重新计算 [0, t-1] 个 token，而是直接计算第 t 个 token
            如果我们内部写 arange(x.seq_len) ，当推理第 t 个 token 的时候，位置就会被错误的当作 t=0 来计算，而不是真正的位置
        左侧 Padding：
            当 batch_size > 1 的时候，训练和推理的不同长度不同，若采用左 padding 对齐，真实的 token 其实位置就不是 0
        Packing：
            训练时把多个短序列拼接成一个长的序列，每条子序列的位置都要各自从0开始
        上下文滑动窗口/长度外推：
            处理超长文本分块、或作位置插值时，需要传入非连续获缩放后的位置
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even * cos - x_odd * sin
        rotated[..., 1::2] = x_even * sin + x_odd * cos
        return rotated
        
        
def scaled_dot_product_attention(
    q: Float[Tensor, "... queries d_heads"],
    k: Float[Tensor, "... keys d_heads"],
    v: Float[Tensor, "... keys d_heads"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
):
        score = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if mask is not None:
            score = score.masked_fill(~mask, -torch.inf)
        return softmax(score, dim=-1) @ v
        
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            max_seq_len: int=None, 
            theta: float=None, 
            device=None, 
            dtype=None
        ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.n_heads = num_heads
        self.d_heads = d_model // num_heads
        self.q_weight = Linear(d_model, d_model, device, dtype)
        self.k_weight = Linear(d_model, d_model, device, dtype)
        self.v_weight = Linear(d_model, d_model, device, dtype)
        self.o_weight = Linear(d_model, d_model, device, dtype)
        if max_seq_len and theta:
            self.rope = RoPE(self.d_heads, theta, max_seq_len, device, dtype)
        else:
            self.rope = None
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: torch.Tensor (batch_size, seq_len, d_model)
        output: torch.Tensor (batch_size, n_heads, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_heads)
        return x.transpose(-3, -2) # 因为每个 haed 都要做一次 seq * seq 的 attention 计算，所以要把 d_head 提前
        
    def forward(self, in_features: Float[Tensor, "... seq_len d_model"], token_positions: torch.Tensor | None = None):
        q = self._split_heads(self.q_weight(in_features)) # (batch, seq, d_model) @ (d_model, d_model) -> split_heads(batch, seq, d_model) -> (batch, n_heads, seq, d_heads)
        k = self._split_heads(self.k_weight(in_features)) # (batch, n_heads, seq, d_heads)
        v = self._split_heads(self.v_weight(in_features)) # (batch, n_heads, seq, d_heads)
        seq_len = in_features.shape[-2]
        
        if self.rope:
            # 这里不旋转V是因为：
            # 如果对 V 也旋转，等于给输出内容凭空叠加了一个与位置相关的旋转扰动，反而破坏了 V 所承载的原始语义信息，且没有任何相对位置抵消机制去还原它。
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=in_features.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool, device=in_features.device)) # (seq, seq)
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_heads) # (batch, n_heads, seq, d_heads) @ (batch, n_heads, d_heads, seq) -> (batch, n_heads, seq, seq)
        causal_attention_score = softmax(attention_score.masked_fill(~causal_mask, torch.finfo(attention_score.dtype).min), dim=-1) # (batch, n_heads, seq, seq)
        attention = causal_attention_score @ v # (batch, n_heads, seq, seq) @ (batch, n_heads, seq, d_heads) -> (batch, n_heads, seq, d_heads)
        
        attention = attention.transpose(-3, -2) # (batch, seq, n_heads, d_heads)
        attention = attention.reshape(*attention.shape[:-2], self.n_heads * self.d_heads) # (batch, seq, d_model)
        return self.o_weight(attention.contiguous()) # (batch, seq, d_model) @ (d_model, d_model) -> (batch, seq, d_model)
    

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, in_features: torch.Tensor, token_positions=None) -> torch.Tensor:
            in_features = in_features + self.mha(self.ln1(in_features), token_positions)
            return in_features + self.ffn(self.ln2(in_features))
        
        
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.embedding_layer = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
       
    def forward(self, in_features: Int[Tensor, "batch_size seq_length"]):
        embedding = self.embedding_layer(in_features) # (batch_size, seq_len, d_model)
        out = embedding
        for layer in self.layers:
            out = layer(out)
        out = self.lm_head(self.ln_final(out))
        return out
