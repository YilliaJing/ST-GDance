from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from model.multihead_diffattn import MultiheadDiffAttn
from torch_geometric.nn import GCNConv

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# LiteLA from the provided code
class LiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        super().__init__()
        heads = heads or int(out_dim // dim * heads_ratio)
        self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=use_bias)
        self.proj = nn.Linear(out_dim, in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # head dimension
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = nn.LayerNorm(in_dim, eps=norm_eps)
            self.k_norm = nn.LayerNorm(in_dim, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = torch.nn.functional.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)
        k = self.k_norm(k).transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)
        v = v.reshape(B, C // self.dim, self.dim, N)

        out = self.attn_matmul(q, k, v).to(dtype)
        out = out.view(B, C, N).permute(0, 2, 1)
        return self.proj(out)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        """
            Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return (self.weight * self._norm(x.float())).type_as(x)

class MultiHeadCrossAttention_new(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            # not used for now
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        _xformers_available = False
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        first_dim = 1 if _xformers_available else B

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(first_dim, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class FiLMTransformerDecoderLayer_sana(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        # Replace nn.MultiheadAttention with LiteLA
        self.self_attn = LiteLA(d_model, d_model, heads=nhead)
        # self.multihead_attn = LiteLA(d_model, d_model, heads=nhead)
        # self.multihead_attn = MultiHeadCrossAttention_new(d_model, nhead, attn_drop=dropout, proj_drop=dropout)  # exp27
        # print(nhead) # 8
        # print(dim_feedforward)  # 1024
        # print(d_model)  # 512
        self.multihead_attn = MultiheadDiffAttn(embed_dim=d_model, depth=0.25, num_heads=nhead)  # exp28
        '''self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )'''

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        return self.self_attn(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        return self.multihead_attn(x, attn_mask=attn_mask)  # self.multihead_attn(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)




class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 400,  # 5 seconds, 30 fps  # 150
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.gnn = GCNConv(latent_dim, latent_dim)
        self.relu = nn.ReLU()
        self.fusion_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),  # 保持特征维度不变
            nn.ReLU(),            # 激活函数
            nn.Linear(latent_dim, latent_dim)
        )

        '''
        ,   # 保持特征维度不变

            nn.Linear(latent_dim, latent_dim),  # 保持特征维度不变
            nn.ReLU(),  # 激活函数
            nn.Linear(latent_dim, latent_dim),  # 保持特征维度不变

            nn.Linear(latent_dim, latent_dim),  # 保持特征维度不变
            nn.ReLU(),  # 激活函数
            nn.Linear(latent_dim, latent_dim)  # 保持特征维度不变
        '''

        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer_sana(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )  # FiLMTransformerDecoderLayer   FiLMTransformerDecoderLayer_sana
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def generate_fully_connected_edge_index(self, num_persons, device):
        # Build a fully connected edge set for each node.
        row = torch.arange(num_persons).repeat(num_persons)  #
        col = torch.arange(num_persons).repeat_interleave(num_persons)

        edge_index = torch.stack([row, col], dim=0).to(device)

        return edge_index

    def generate_fully_connected_edge_index_with_threshold_batch(self, trans, device, top_k=0.2):
        """
        **Calculate edge_index and edge_weight for all batches at once**
        :param trans: (batch_size, num_persons, 3) - The xy position of each person in each batch
        :param device: The computing device (CPU/GPU)
        :param top_k: The maximum edge weight ratio selected (default 20%)
        :return: edge_index (2, batch_size * num_edges), edge_weight (batch_size * num_edges)
        """
        batch_size, num_persons, _ = trans.shape

        # graph constrution
        row = torch.arange(num_persons).repeat(num_persons)
        col = torch.arange(num_persons).repeat_interleave(num_persons)

        batch_row = row.repeat(batch_size, 1)  # (batch_size, num_edges)
        batch_col = col.repeat(batch_size, 1)  # (batch_size, num_edges)
        trans_xy = trans[:, :, :2]
        batch_edge_weight = torch.norm(trans_xy[:, row] - trans_xy[:, col], dim=2)  # (batch_size, num_edges)

        threshold = torch.quantile(batch_edge_weight, 1 - top_k, dim=1, keepdim=True)  # (batch_size, 1)
        mask = batch_edge_weight >= threshold  # (batch_size, num_edges)

        
        batch_row = batch_row[mask].flatten()
        batch_col = batch_col[mask].flatten()
        batch_edge_weight = batch_edge_weight[mask].flatten()

        batch_edge_weight = F.normalize(batch_edge_weight, p=2, dim=0)

        
        edge_index = torch.stack([batch_row, batch_col], dim=0).to(device)
        edge_weight = batch_edge_weight.to(device)

        return edge_index, edge_weight

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device  # batch_size, max_person, target_seq, dims=151=3+24*6+4   trans,poses,contact
        if x.dtype != self.input_projection.weight.dtype:
            x = x.to(self.input_projection.weight.dtype)  # 确保输入数据类型与权重一致'''
        # project to latent space
        x = self.input_projection(x)

        edge_index = self.generate_fully_connected_edge_index(num_persons=3, device=device)  # 改这里
        gnn_out = self.relu(self.gnn(x, edge_index))  # [32, 5, 150, 512]

        #---------- new GNN：---------
        trans = x[:, :, 0, :2]  # x
        batch_size, num_persons, _ = trans.shape

        edge_index, edge_weight = self.generate_fully_connected_edge_index_with_threshold_batch(
            trans, device, top_k=0.2
        )
        
        x = x.view(-1, x.shape[-1])  # (batch_size * num_persons, feature_dim)

        gnn_out = self.relu(self.gnn(x, edge_index, edge_weight))

        gnn_out = gnn_out.view(batch_size, num_persons, -1)

        # add Fusion Projection  
        bn, num_persons, frames, hidden_dim = gnn_out.shape
        x = gnn_out.reshape(bn, num_persons * frames, hidden_dim)
        x = self.fusion_projection(x)

        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden  # ~

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)  # 128,5*150,512 ->

        output = self.final_layer(output)

        output = output.reshape(bn, num_persons, frames, -1)
        return output   # bn, num_persons, frames, 151
