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

"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return torch.tensor(float('-inf'), dtype=dtype)


# 辅助函数：创建解码器自注意力机制的掩码 (causal mask)
# 这个函数确保解码器在预测当前位置时不能看到未来的信息。
# 输出的 mask 中，True 代表“保留”，False 代表“遮盖未来信息”。
# 在 MultiHeadAttention 中，会通过 (mask == 0) 来获取真正要填充 -inf 的位置。
def _create_selfattn_mask(target_tensor: torch.Tensor) -> torch.Tensor:
    """
    为解码器的自注意力机制创建因果掩码 (causal mask)。
    防止注意力机制看到未来的 token。

    参数:
        target_tensor (torch.Tensor): 目标序列张量，形状为 (batch_size, seq_len, dim)。

    返回:
        torch.Tensor: 因果掩码张量，形状为 (batch_size, seq_len, seq_len)。
                      值为 True 的位置表示允许注意力，值为 False 的位置表示禁止注意力。
                      在MHA中，通常 (mask == 0) 的位置会被填充为负无穷。
                      所以这里 True 表示看到，False表示看不到。
                      MHA内部 (mask == 0) 后，看不到的地方会变成True，然后被masked_fill_。
    """
    batch_size, seq_len, _ = target_tensor.size()
    # 创建一个下三角矩阵 (包括对角线)，值为 True。上三角为 False。
    # True 表示 token i 可以关注 token j (j<=i)
    # False 表示 token i 不可以关注 token j (j>i)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=target_tensor.device, dtype=torch.bool))
    # 扩展到 batch_size
    return mask.unsqueeze(0).expand(batch_size, seq_len, seq_len)


def create_position_codes(n_pos: int, dim: int, out: torch.Tensor):
    """
    创建正弦/余弦位置编码。

    参数:
        n_pos: 位置数量 (最大序列长度)。
        dim: 嵌入维度。
        out: 用于存储位置编码的张量 (通常是 nn.Embedding.weight)。
    """
    # 验证 dim 是否为偶数，因为位置编码成对计算 (sin, cos)
    assert dim % 2 == 0, "Embedding dimension must be even for sinusoidal position codes."

    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])  # shape: (n_pos, dim / 2)

    # out 是一个预先分配的张量，例如 nn.Embedding.weight
    # 将计算出的正弦值赋给偶数索引列
    out.data[:, 0::2] = torch.from_numpy(np.sin(position_enc)).float()
    # 将计算出的余弦值赋给奇数索引列
    out.data[:, 1::2] = torch.from_numpy(np.cos(position_enc)).float()

    # 从计算图中分离，因为这些编码是固定的，不是通过梯度下降学习的 (除非 learn_positional_embeddings=True)
    out.detach_()
    # 设置为不需要梯度
    out.requires_grad = False


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)

class MultiHeadAttention(nn.Module):
    """
    标准的多头注意力机制模块。
    """
    def __init__(self, n_heads: int, dim: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads  # 注意力头的数量
        self.dim = dim          # 输入和输出的维度

        # 确保维度可以被头的数量整除
        assert dim % n_heads == 0, "dim必须能被n_heads整除"
        self.dim_per_head = dim // n_heads # 每个头的维度

        # 注意力分数计算后的 Dropout
        self.attn_dropout = nn.Dropout(p=dropout)

        # Query, Key, Value 的线性变换层
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)

        # 权重初始化 (Xavier Normal)
        # nn.Linear 默认会初始化偏置（如果bias=True, 默认为True）。
        # 对于偏置，通常初始化为0，nn.Linear默认的uniform初始化接近0。
        # 如果想精确设为0：
        # if self.q_lin.bias is not None: nn.init.zeros_(self.q_lin.bias)
        # ...以此类推
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)

        # 输出前的线性变换层
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)
        # if self.out_lin.bias is not None: nn.init.zeros_(self.out_lin.bias)

        # 然后在forward中分割，这有时能提高参数加载和计算效率。
        # 但当前分开定义更清晰。

    def _prepare_head(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将输入张量调整为多头注意力的形式。
        输入: (batch_size, seq_len, n_heads * dim_per_head)
        输出: (batch_size * n_heads, seq_len, dim_per_head)
        """
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.n_heads, self.dim_per_head)
        # (batch_size, n_heads, seq_len, dim_per_head)
        tensor = tensor.transpose(1, 2).contiguous()
        # (batch_size * n_heads, seq_len, dim_per_head)
        tensor = tensor.view(batch_size * self.n_heads, seq_len, self.dim_per_head)
        return tensor

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            query (torch.Tensor): Query张量, 形状 (batch_size, query_len, dim).
            key (Optional[torch.Tensor]): Key张量, 形状 (batch_size, key_len, dim).
                                          如果为None, 则为自注意力 (key=query).
            value (Optional[torch.Tensor]): Value张量, 形状 (batch_size, key_len, dim).
                                            如果为None, 则为自注意力 (value=query or value=key).
            mask (Optional[torch.Tensor]): 掩码张量。
                - 对于自注意力中的因果掩码: 形状通常是 (batch_size, query_len, key_len),
                  其中 True 表示允许注意力，False 表示禁止（将被填充为-inf）。
                - 对于padding掩码: 形状通常是 (batch_size, key_len) or (batch_size, 1, key_len),
                  其中 1 (True) 表示有效token, 0 (False) 表示padding token。
                  (mask == 0) 会被用于标记需要填充 -inf 的位置。

        返回:
            torch.Tensor: 注意力机制的输出, 形状 (batch_size, query_len, dim).
        """
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'输入张量的维度 ({dim}) 与模块配置的维度 ({self.dim}) 不匹配'
        # mask 在实际应用中通常是必须的，尤其是对于padding或因果自注意力
        # assert mask is not None, 'Mask 不能为 None, 请指定一个有效的 mask'
        # 根据原代码，mask可能为None，这里注释掉强制检查，但推荐始终提供mask

        # 计算缩放因子
        scale = math.sqrt(self.dim_per_head)

        # 1. 处理自注意力的情况 (key/value 未提供)
        if key is None and value is None:
            # 自注意力机制
            key = value = query
        elif value is None:
            # 通常意味着 key 和 value 相同 (例如 encoder-decoder attention 中 encoder_output作为K和V)
            value = key

        # 断言 key 和 value 不为空 (经过上面的处理后)
        assert key is not None and value is not None
        _, key_len, _ = key.size() # 获取 key 的序列长度

        # 2. 线性变换并调整形状以适应多头
        q = self.q_lin(query)       # (batch_size, query_len, dim)
        k = self.k_lin(key)         # (batch_size, key_len, dim)
        v = self.v_lin(value)       # (batch_size, key_len, dim)

        q_prepared = self._prepare_head(q)  # (batch_size * n_heads, query_len, dim_per_head)
        k_prepared = self._prepare_head(k)  # (batch_size * n_heads, key_len, dim_per_head)
        v_prepared = self._prepare_head(v)  # (batch_size * n_heads, key_len, dim_per_head)

        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # (batch_size * n_heads, query_len, dim_per_head) @ (batch_size * n_heads, dim_per_head, key_len)
        # -> (batch_size * n_heads, query_len, key_len)
        dot_prod = torch.bmm(q_prepared, k_prepared.transpose(1, 2)) / scale

        # 4. 应用掩码 (如果提供)
        if mask is not None:
            # attn_mask 的形状需要是 (batch_size * n_heads, query_len, key_len)
            # mask_fill_ 的条件是 True 的地方被填充。
            # 我们希望 padding (mask中为0) 或未来token (causal mask中为0或False) 的位置被填充。
            # 所以条件是 (mask_input == 0) 或 (mask_input == False)

            # 处理 padding mask (B, K_len) 或 (B, 1, K_len)
            if mask.dim() == 2: # (B, K_len)
                attn_mask_logical = (mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, K_len)
            # 处理 causal mask or combined mask (B, Q_len, K_len)
            elif mask.dim() == 3: # (B, Q_len, K_len)
                attn_mask_logical = (mask == 0).unsqueeze(1) # (B, 1, Q_len, K_len)
            else:
                raise ValueError(f"不支持的mask维度: {mask.shape}")

            # 扩展到多头和query_len (如果需要)
            # (B, 1, Q_len for causal or 1 for padding, K_len) -> (B, H, Q_len, K_len)
            attn_mask_expanded = attn_mask_logical.expand(batch_size, self.n_heads, query_len, key_len)
            attn_mask_final = attn_mask_expanded.reshape(batch_size * self.n_heads, query_len, key_len)

            assert attn_mask_final.shape == dot_prod.shape, \
                f"掩码形状 {attn_mask_final.shape} 与注意力分数形状 {dot_prod.shape} 不匹配"
            dot_prod.masked_fill_(attn_mask_final, neginf(dot_prod.dtype))

        # 5. 计算注意力权重 (Softmax)
        attn_weights = F.softmax(dot_prod, dim=-1) # 在 key_len 维度上 softmax
        attn_weights = self.attn_dropout(attn_weights) # 应用 dropout

        # 6. 加权求和 Value
        # (batch_size * n_heads, query_len, key_len) @ (batch_size * n_heads, key_len, dim_per_head)
        # -> (batch_size * n_heads, query_len, dim_per_head)
        attentioned = torch.bmm(attn_weights, v_prepared)

        # 7. 恢复形状
        # (batch_size * n_heads, query_len, dim_per_head) -> (batch_size, n_heads, query_len, dim_per_head)
        attentioned = attentioned.view(batch_size, self.n_heads, query_len, self.dim_per_head)
        # (batch_size, query_len, n_heads, dim_per_head)
        attentioned = attentioned.transpose(1, 2).contiguous()
        # (batch_size, query_len, dim)
        attentioned = attentioned.view(batch_size, query_len, self.dim)

        # 8. 输出前的线性变换
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


class StandardPositionalEncoding(nn.Module):
    """
    标准的Transformer位置编码模块。
    既可以处理可学习的位置编码，也可以处理固定的正弦/余弦位置编码。

    :param int embedding_size: 词嵌入的维度 (d_model)。
    :param int n_positions: 最大序列长度 (max_len)。
    :param bool learnable: 如果为True，则位置编码是可学习的参数；
                                 如果为False，则使用固定的正弦/余弦编码。
    """

    def __init__(self, embedding_size: int, n_positions: int = 1024, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        self.embedding_size = embedding_size
        self.n_positions = n_positions

        if self.learnable:
            # 如果是可学习的位置编码，创建一个 Embedding 层作为查找表
            self.embedding = nn.Embedding(self.n_positions, self.embedding_size)
            # 使用均值为0，标准差为 embedding_size^-0.5 的正态分布初始化
            nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding_size ** -0.5)
        else:
            # 如果是固定的正弦/余弦编码，预先计算权重并注册为缓冲区
            sinusoidal_weights = self._get_sinusoidal_embeddings()
            # register_buffer 会将张量注册到模块，使其可以被 state_dict 追踪，
            # 并且会自动移动到正确的设备 (cpu/gpu)，但不会被视为模型参数。
            self.register_buffer('sinusoidal_weights', sinusoidal_weights)
            # 在这种情况下，我们不需要一个 nn.Embedding 层
            self.embedding = None # 明确表示不使用 nn.Embedding

    def _get_sinusoidal_embeddings(self) -> torch.Tensor:
        """
        生成正弦/余弦位置编码。
        参考 "Attention Is All You Need" 论文中的公式。
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        输出形状: [n_positions, embedding_size]
        """
        d_model = self.embedding_size
        # 初始化一个形状为 [n_positions, d_model] 的零张量，用于存放位置编码
        pe = torch.zeros(self.n_positions, d_model)

        # 创建一个表示位置的张量，形状为 [n_positions, 1] (例如 [[0], [1], ..., [n_positions-1]])
        position = torch.arange(0, self.n_positions, dtype=torch.float).unsqueeze(1)

        # 计算除法项 div_term，对应于公式中的 1 / (10000^(2i/d_model))
        # _2i 对应公式中的 2i，从 0 开始，步长为 2
        # div_term 的长度将是 ceil(d_model / 2)
        _2i = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model))

        # 计算偶数索引位置(0, 2, 4, ...)的正弦编码
        # pe[:, 0::2] 会选择 ceil(d_model / 2) 列
        pe[:, 0::2] = torch.sin(position * div_term)

        # 计算奇数索引位置(1, 3, 5, ...)的余弦编码
        # pe[:, 1::2] 会选择 floor(d_model / 2) 列
        # 因此，对于 div_term，我们只需要其前 d_model // 2 个元素来与这些列相乘
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        return pe

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        :param positions: 形状为 [batch_size, seq_len] 的张量，包含位置索引
                          (整数，范围从 0 到 n_positions-1)。
        :return: 形状为 [batch_size, seq_len, embedding_size] 的位置编码张量。
        """
        if self.learnable:
            # 如果是可学习的，通过 embedding 层查找对应位置的编码
            # 确保 self.embedding 不为 None (在 learnable=True 时被初始化)
            if self.embedding is None:
                raise RuntimeError("Positional encoding is learnable, but nn.Embedding layer was not initialized.")
            return self.embedding(positions)
        else:
            # 如果是固定的，直接从预计算的 sinusoidal_weights 中提取
            # sinusoidal_weights 的形状是 [n_positions, embedding_size]
            # positions 的形状是 [batch_size, seq_len]
            # 我们期望输出形状是 [batch_size, seq_len, embedding_size]
            # 直接使用 tensor indexing 即可实现
            if self.sinusoidal_weights is None: # 理论上不应该发生，因为 __init__ 会初始化它
                 raise RuntimeError("Positional encoding is fixed, but sinusoidal_weights buffer was not initialized.")
            return self.sinusoidal_weights[positions]


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

        self.positional_encoder = StandardPositionalEncoding(
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
