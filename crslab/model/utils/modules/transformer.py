# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import math
from typing import Optional, Union, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PositionalEncoding

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def _create_selfattn_mask(x):
    # figure out how many timestamps we need
    bsz = x.size(0)
    time = x.size(1)
    # make sure that we don't look into the future
    mask = torch.tril(x.new(time, time).fill_(1))
    # broadcast across batch
    mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    return mask


def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out.data[:, 0::2] = torch.as_tensor(np.sin(position_enc))
    out.data[:, 1::2] = torch.as_tensor(np.cos(position_enc))
    out.detach_()
    out.requires_grad = False


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
                .view(batch_size, 1, -1, key_len)
                .repeat(1, n_heads, 1, 1)
                .expand(batch_size, n_heads, query_len, key_len)
                .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2).contiguous()
                .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out



class TransformerFFN(nn.Module):
    """
    Transformer 模型中的前馈网络 (Feed-Forward Network) 模块。

    它通常由两个线性层和一个中间的非线性激活函数组成。
    FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    Dropout 可以在激活之后、第二个线性层之前应用。
    """
    def __init__(self,
                 dim: int,                       # 输入和输出维度
                 dim_hidden: int,                # 隐藏层维度
                 activation_fn_class: Type[nn.Module] = nn.ReLU, # 激活函数类 (例如 nn.ReLU, nn.GELU)
                 activation_dropout: float = 0.0, # 在激活函数和第二个线性层之间的 dropout 比率
                 use_bias: bool = True           # 线性层是否使用偏置项
                ):
        super(TransformerFFN, self).__init__()
        self.dim = dim
        self.dim_hidden = dim_hidden

        # 如果 dropout 为 0，使用 nn.Identity() 更清晰地表示无操作
        self.activation_dropout_layer = nn.Dropout(p=activation_dropout) if activation_dropout > 0.0 else nn.Identity()

        self.lin1 = nn.Linear(dim, dim_hidden, bias=use_bias)
        self.activation = activation_fn_class() # 实例化激活函数
        self.lin2 = nn.Linear(dim_hidden, dim, bias=use_bias)

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重和偏置。"""
        nn.init.xavier_uniform_(self.lin1.weight)
        if self.lin1.bias is not None:
            nn.init.zeros_(self.lin1.bias)

        nn.init.xavier_uniform_(self.lin2.weight)
        if self.lin2.bias is not None:
            nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        :param x: 输入张量，形状为 (..., dim)
        :return: 输出张量，形状为 (..., dim)
        """
        x = self.lin1(x)
        x = self.activation(x)
        x = self.activation_dropout_layer(x) # 应用激活后的 dropout
        x = self.lin2(x)
        return x


class TransformerEncoderLayer(nn.Module): # 保持与之前一致的简化版
    def __init__(self, n_heads: int, embedding_size: int, ffn_size: int,
                 attention_dropout: float, relu_dropout: float, dropout: float,
                 activation_fn: nn.Module = nn.ReLU()): # 新增激活函数参数
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_size, n_heads, dropout=attention_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_size, ffn_size),
            activation_fn, # 使用可配置的激活函数
            nn.Dropout(relu_dropout),
            nn.Linear(ffn_size, embedding_size),
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 改进 dropout
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 改进 dropout

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: True 表示 padding, MultiheadAttention 的期望格式
        attended_x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attended_x))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x



# --- 改进后的 TransformerEncoder ---
class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_dropout: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions: Size of the position embeddings matrix.
    """
    def __init__(
            self,
            n_heads: int,
            n_layers: int,
            embedding_size: int,
            ffn_size: int,
            embedding: Optional[nn.Embedding] = None,
            vocabulary_size: Optional[int] = None, # 改为可选
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            relu_dropout: float = 0.0,
            padding_idx: int = 0,
            learn_positional_embeddings: bool = False,
            embeddings_scale: bool = False,
            reduction: bool = True,
            n_positions: int = 1024,
            ffn_activation_fn: nn.Module = nn.ReLU() # 新增 FFN 激活函数
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity() # 改进 dropout
        self.out_dim = embedding_size

        if embedding_size % n_heads != 0:
            raise ValueError(
                f"Transformer embedding_size ({embedding_size}) "
                f"must be a multiple of n_heads ({n_heads})"
            )

        if embedding is not None:
            if embedding.embedding_dim != embedding_size:
                 raise ValueError(
                    f"Provided embedding dim ({embedding.embedding_dim}) "
                    f"must match the embedding_size ({embedding_size})."
                )
            self.embeddings = embedding
        else:
            if vocabulary_size is None:
                raise ValueError(
                    "vocabulary_size must be provided if embedding is None."
                )
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, mean=0, std=embedding_size ** -0.5)

        self.positional_encoder = PositionalEncoding(
            embedding_size, n_positions, learn_positional_embeddings
        )

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout, # 注意这里传递的是原始的 dropout 值
                activation_fn=ffn_activation_fn # 传递激活函数
            ))

    def forward(self, input_ids: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param input_ids: 形状为 [batch, seq_len] 的 LongTensor
        :return: 如果 reduction=True, 返回 [batch, embedding_size] 的 Tensor。
                 如果 reduction=False, 返回 ([batch, seq_len, embedding_size], [batch, seq_len]) 的 Tuple。
        """
        # input_ids (batch_size, seq_len)
        # key_padding_mask: (batch_size, seq_len), True 表示是 padding
        # attention_mask (for MultiheadAttention): (batch_size, seq_len), True 表示要被 MASK (即 padding)
        input_mask = (input_ids == self.padding_idx) # True for padding positions

        # positions: (batch_size, seq_len)
        # 非 padding 位置从 0 开始计数，padding 位置的值可能不重要，因为会被 mask
        non_pad_mask = ~input_mask # True for non-padding
        positions = (non_pad_mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        positions.masked_fill_(input_mask, 0) # 将 padding 位置的 position 设为0 (或任何有效索引)

        tensor = self.embeddings(input_ids)
        if self.embeddings_scale:
            tensor = tensor * (self.embedding_size ** 0.5) # 通常乘以维度的平方根

        tensor = tensor + self.positional_encoder(positions)
        tensor = self.dropout(tensor)

        # 将 padding 位置的 tensor 值置为 0 (可选，因为注意力掩码会处理)
        # 但为了后续的 sum/mean pooling 的正确性，这里处理是好的
        tensor.masked_fill_(input_mask.unsqueeze(-1), 0.0)

        for layer in self.layers:
            # TransformerEncoderLayer 的 forward 需要 key_padding_mask
            # MultiheadAttention 的 key_padding_mask 中 True 代表 padding
            tensor = layer(tensor, key_padding_mask=input_mask)

        if self.reduction:
            # (batch, seq_len, dim) -> (batch, dim)
            # 只对非 padding 部分进行平均
            num_non_padding = non_pad_mask.sum(dim=1, keepdim=True).float().clamp(min=1e-7) # (batch, 1)
            output = tensor.sum(dim=1) / num_non_padding
            return output
        else:
            # (batch, seq_len, dim), (batch, seq_len)
            return tensor, non_pad_mask # 返回 non_pad_mask 可能比原始 input_mask 更有用


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding=None,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            embeddings_scale=True,
            learn_positional_embeddings=False,
            padding_idx=None,
            n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.shape[1]
        positions = input.new_empty(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (batch, seq_len)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None
