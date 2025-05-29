import numpy as np
import torch
from torch import nn as nn

from crslab.model.utils.modules.transformer import MultiHeadAttention, TransformerFFN, _create_selfattn_mask, \
    _normalize, \
    create_position_codes


class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb

from typing import Optional, Type

class GateLayerImproved(nn.Module):
    """
    一个改进的门控层 (Gate Layer)，它根据学习到的门控机制动态地结合两个输入张量。

    门控信号由两个输入张量的拼接结果计算得出，该结果会依次通过两个线性层和
    一个 sigmoid 激活函数。在门控计算路径的第一个和第二个线性层之间，
    可以选择性地应用一个中间激活函数。
    """
    def __init__(self,
                 input_dim: int,
                 intermediate_activation: Optional[Type[nn.Module]] = nn.ReLU):
        """
        初始化 GateLayerImproved。

        参数:
            input_dim (int): 两个输入张量 (input1, input2) 最后一个维度的特征数。
            intermediate_activation (Optional[Type[nn.Module]], 可选):
                在门控计算路径中，第一个线性层之后应用的激活函数类。
                如果为 None，则不应用任何中间激活函数 (等效于 nn.Identity)。
                默认为 nn.ReLU。 例如: nn.ReLU, nn.Tanh。
        """
        super().__init__() # 使用现代的 super() 写法
        self.input_dim = input_dim

        # 用于计算门控信号的层
        # 第一个线性层，处理拼接后的输入
        self.gate_fc1 = nn.Linear(input_dim * 2, input_dim)

        # 中间激活函数
        if intermediate_activation is not None:
            self.gate_intermediate_activation = intermediate_activation()
        else:
            self.gate_intermediate_activation = nn.Identity() # 如果不指定，则不进行操作

        # 第二个线性层，输出原始门控值 (gate logits)
        self.gate_fc2 = nn.Linear(input_dim, 1)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        定义 GateLayerImproved 的前向传播过程。

        参数:
            input1 (torch.Tensor): 第一个输入张量。
                                   期望形状: (..., input_dim)
            input2 (torch.Tensor): 第二个输入张量。
                                   期望形状: (..., input_dim)
                                   必须与 input1 具有相同的形状以便进行加权求和。

        返回:
            torch.Tensor: input1 和 input2 经过门控机制融合后的张量。
                          形状: (..., input_dim)
        """
        # 确保输入维度符合预期 (虽然线性层本身也会检查，但显式说明有助于理解)
        # 为简洁起见，这里依赖 PyTorch 内部的维度检查

        # 1. 拼接输入
        concatenated_inputs = torch.cat([input1, input2], dim=-1)

        # 2. 计算门控信号
        # 通过第一个全连接层
        gate_hidden = self.gate_fc1(concatenated_inputs)
        # 应用中间激活函数
        gate_hidden_activated = self.gate_intermediate_activation(gate_hidden)
        # 通过第二个全连接层得到原始门控值
        gate_raw_output = self.gate_fc2(gate_hidden_activated)
        # 应用 sigmoid 函数得到范围在 (0, 1) 的门控信号
        gate = torch.sigmoid(gate_raw_output)  # 形状: (..., 1)

        # 3. 应用门控机制
        # PyTorch 的广播机制会自动处理 gate (..., 1) 与 input1/input2 (..., input_dim) 的乘法
        gated_output = gate * input1 + (1 - gate) * input2

        return gated_output


class TransformerDecoderLayerKG(nn.Module):
    """
    带知识图谱（KG）和数据库（DB）集成的 Transformer 解码器层。

    该层包含：
    1. 自注意力机制 (self-attention)
    2. 与数据库编码器输出的交叉注意力机制 (encoder-db attention)
    3. 与知识图谱编码器输出的交叉注意力机制 (encoder-kg attention)
    4. 与标准编码器输出的交叉注意力机制 (encoder attention)
    5. 前馈网络 (feed-forward network)

    每个子层后都跟着 Dropout 和 Add & Norm (残差连接 + Layer Normalization)。
    """
    def __init__(
        self,
        n_heads: int,
        embedding_size: int,
        ffn_size: int,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0, # 通常称为 activation_dropout 或 ffn_dropout
        dropout: float = 0.0,      # 残差连接后的dropout
    ):
        super().__init__()
        self.dim = embedding_size  # 嵌入维度
        self.ffn_dim = ffn_size    # FFN中间层的维度

        # Dropout 层，用于残差连接之后
        self.dropout = nn.Dropout(p=dropout)

        # 1. 自注意力模块
        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        # 2. 与DB编码器输出的交叉注意力模块
        self.encoder_db_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_db = nn.LayerNorm(embedding_size)

        # 3. 与KG编码器输出的交叉注意力模块
        self.encoder_kg_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_kg = nn.LayerNorm(embedding_size)

        # 4. 与标准编码器输出的交叉注意力模块
        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        # 5. 前馈网络模块
        self.ffn = TransformerFFN(embedding_size, ffn_size, activation_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(
        self,
        x: torch.Tensor,                          # 解码器输入, 形状: (batch_size, target_seq_len, embedding_size)
        encoder_output: torch.Tensor,             # 标准编码器输出, 形状: (batch_size, source_seq_len, embedding_size)
        encoder_mask: torch.Tensor,               # 标准编码器掩码, 形状: (batch_size, 1, source_seq_len) or (batch_size, source_seq_len)
        kg_encoder_output: torch.Tensor,          # KG编码器输出, 形状: (batch_size, kg_seq_len, embedding_size)
        kg_encoder_mask: torch.Tensor,            # KG编码器掩码, 形状: (batch_size, 1, kg_seq_len) or (batch_size, kg_seq_len)
        db_encoder_output: torch.Tensor,          # DB编码器输出, 形状: (batch_size, db_seq_len, embedding_size)
        db_encoder_mask: torch.Tensor,            # DB编码器掩码, 形状: (batch_size, 1, db_seq_len) or (batch_size, db_seq_len)
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 解码器输入。
            encoder_output: 标准编码器的输出。
            encoder_mask: 标准编码器输出的掩码 (通常是padding mask, 1代表有效token, 0代表padding)。
            kg_encoder_output: 知识图谱编码器的输出。
            kg_encoder_mask: 知识图谱编码器输出的掩码。
            db_encoder_output: 数据库编码器的输出。
            db_encoder_mask: 数据库编码器输出的掩码。
        返回:
            torch.Tensor: 解码器层的输出, 形状与x相同。
        """
        # --- 1. 自注意力机制 ---
        residual = x
        # 创建自注意力掩码，防止看到未来的信息
        # _create_selfattn_mask 返回的mask中, True表示可见, False表示遮蔽
        # MultiHeadAttention内部会处理 (mask == 0) 的情况
        decoder_mask = _create_selfattn_mask(x) # 形状: (batch_size, target_seq_len, target_seq_len)

        # self_attention 期望 mask 的形状是 [B, Q_len, K_len]
        # 其中被遮蔽的位置（例如未来token或padding）应在MHA内部计算得到 True (mask_fill用)
        # 如果 decoder_mask 中 False 代表遮蔽, MHA内部 (decoder_mask == False) 即 (decoder_mask == 0)
        # 会将这些位置标记为 True，然后用 -inf 填充。
        x_attn = self.self_attention(query=x, key=x, value=x, mask=decoder_mask) # Q,K,V相同；使用因果掩码
        x = self.dropout(x_attn)
        x = x + residual
        x = self.norm1(x)

        # --- 2. 与DB编码器的交叉注意力 ---
        residual = x
        # encoder_db_mask: (B, db_seq_len) or (B, 1, db_seq_len)
        # MHA内部会正确处理padding (mask中0代表padding, 会被MHA转为True来mask_fill)
        x_attn = self.encoder_db_attention(
            query=x,
            key=db_encoder_output,
            value=db_encoder_output,
            mask=db_encoder_mask  # 使用DB编码器的padding mask
        )
        x = self.dropout(x_attn)
        x = residual + x
        x = self.norm2_db(x)

        # --- 3. 与KG编码器的交叉注意力 ---
        residual = x
        x_attn = self.encoder_kg_attention(
            query=x,
            key=kg_encoder_output,
            value=kg_encoder_output,
            mask=kg_encoder_mask # 使用KG编码器的padding mask
        )
        x = self.dropout(x_attn)
        x = residual + x
        x = self.norm2_kg(x)

        # --- 4. 与标准编码器的交叉注意力 ---
        residual = x
        x_attn = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask     # 使用标准编码器的padding mask
        )
        x = self.dropout(x_attn)
        x = residual + x
        x = self.norm2(x)

        # --- 5. 前馈网络 ---
        residual = x
        x_ffn = self.ffn(x)
        x = self.dropout(x_ffn)
        x = residual + x
        x = self.norm3(x)

        return x


class TransformerDecoderKG(nn.Module):
    """
    集成了知识图谱（KG）和数据库（DB）信息的 Transformer 解码器。
    这个模块将堆叠多个 TransformerDecoderLayerKG 层。

    参数:
        n_heads (int): 多头注意力的头数。
        n_layers (int): TransformerDecoderLayerKG 的层数。
        embedding_size (int): 嵌入维度。必须是 n_heads 的倍数。
        ffn_size (int): 前馈网络 (FFN) 中隐藏层的大小。
        vocabulary_size (int): 词汇表大小。 (如果提供了 embedding 参数，这个可能主要用于参考或输出层)。
        embedding (nn.Embedding): 词嵌入层。如果为 None，则理论上应该在此模块内创建一个。
                                  (当前代码直接使用传入的 embedding)。
        dropout (float): 应用于嵌入和层归一化之前的 Dropout 率。
                         (遵循 Vaswani et al., 2017 的用法)。
        attention_dropout (float): 多头注意力中 softmax 之后应用的 Dropout 率。
                                   (Vaswani et al., 2017 未使用)。
        relu_dropout (float): FFN 中 ReLU (或等效激活函数) 之后应用的 Dropout 率。
                              (Vaswani et al., 2017 未使用, 但 Tensor2Tensor 中使用)。
        embeddings_scale (bool): 是否根据嵌入维度缩放嵌入向量 (乘以 sqrt(dim))。
                                 (在 fairseq 中发现有用)。
        learn_positional_embeddings (bool): 如果为 True，则位置嵌入从头开始学习。
                                            如果为 False，则使用固定的正弦/余弦位置嵌入。
        padding_idx (int, optional): 嵌入矩阵中保留的填充索引。如果词嵌入层已配置此项，则此处为参考。
        n_positions (int): 位置嵌入矩阵的大小 (即模型能处理的最大序列长度)。
    """

    def __init__(
            self,
            n_heads: int,
            n_layers: int,
            embedding_size: int,
            ffn_size: int,
            vocabulary_size: int,  # 注意：如果 embedding 是外部传入的，此参数可能仅作记录
            embedding: nn.Embedding,  # 通常是预训练或共享的词嵌入层
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            relu_dropout: float = 0.0,
            embeddings_scale: bool = True,
            learn_positional_embeddings: bool = False,
            padding_idx: int = None,  # 通常词嵌入层会处理 padding_idx
            n_positions: int = 1024,  # 最大序列长度
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size  # 嵌入维度，与 embedding_size 相同
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(dropout)  # 通用 Dropout 层

        self.out_dim = embedding_size  # 输出维度与嵌入维度一致
        assert embedding_size % n_heads == 0, \
            'Transformer 的嵌入维度必须是注意力头数 (n_heads) 的整数倍'

        # 词嵌入层
        # 假设传入的 embedding 是一个 nn.Embedding 实例
        self.embeddings = embedding
        # padding_idx 应该在创建 self.embeddings 时指定，例如:
        # self.embeddings = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_idx)
        # 如果 embedding 是外部传入的，它应该已经配置好了 padding_idx (如果需要)

        # 位置嵌入层
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            # 如果不学习位置嵌入，则创建固定的正弦/余弦位置编码
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            # 如果学习位置嵌入，则进行正态初始化
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # 构建解码器层
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayerKG(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,  # 传递给 TransformerDecoderLayerKG 内部的 FFN
                dropout=dropout,  # 传递给 TransformerDecoderLayerKG 用于残差连接后的dropout
            ))

    def forward(self,
                input_tokens: torch.Tensor,  # 输入的token IDs, 形状: (batch_size, target_seq_len)
                encoder_state: tuple,  # 编码器状态，包含编码器输出和掩码
                kg_encoder_output: torch.Tensor,  # KG编码器输出, 形状: (batch_size, kg_seq_len, embedding_size)
                kg_encoder_mask: torch.Tensor,  # KG编码器掩码
                db_encoder_output: torch.Tensor,  # DB编码器输出, 形状: (batch_size, db_seq_len, embedding_size)
                db_encoder_mask: torch.Tensor,  # DB编码器掩码
                incremental_state=None  # 用于增量解码（推理时优化），此处未完全实现
                ):
        # 从 encoder_state 中解包标准编码器的输出和掩码
        encoder_output, encoder_mask = encoder_state  # encoder_output: (B, src_len, D), encoder_mask: (B, 1, src_len) or (B, src_len)

        target_seq_len = input_tokens.size(1)

        # 1. 创建位置 IDs
        # positions 张量形状: (target_seq_len) -> (1, target_seq_len)
        positions = input_tokens.new_empty(target_seq_len, dtype=torch.long)  # 创建一个与input_tokens相同设备和类型的空张量
        positions = torch.arange(target_seq_len, out=positions).unsqueeze(
            0)  # (1, target_seq_len), 内容为 [0, 1, ..., target_seq_len-1]

        # 2. 获取词嵌入
        # embedded_tokens 形状: (batch_size, target_seq_len, embedding_size)
        embedded_tokens = self.embeddings(input_tokens)

        # 3. (可选) 缩放词嵌入
        if self.embeddings_scale:
            # 按照 fairseq 的做法，将嵌入乘以其维度的平方根
            embedded_tokens = embedded_tokens * np.sqrt(self.dim)

        # 4. 添加位置嵌入
        # position_embeddings(positions) 形状: (1, target_seq_len, embedding_size)
        # expand_as(embedded_tokens) 将其扩展为 (batch_size, target_seq_len, embedding_size)
        # tensor 形状: (batch_size, target_seq_len, embedding_size)
        tensor = embedded_tokens + self.position_embeddings(positions).expand_as(embedded_tokens)

        # 5. 应用 Dropout
        tensor = self.dropout(tensor)  # 这是 Vaswani 2017 中的 dropout，在输入到 Transformer 层之前

        # 6. 逐层通过 TransformerDecoderLayerKG
        for layer in self.layers:
            tensor = layer(
                x=tensor,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                kg_encoder_output=kg_encoder_output,
                kg_encoder_mask=kg_encoder_mask,
                db_encoder_output=db_encoder_output,
                db_encoder_mask=db_encoder_mask
            )

        # 返回最终的解码器输出和未来的增量状态 (此处为None)
        # tensor 形状: (batch_size, target_seq_len, embedding_size)
        return tensor, None  # `None` 对应于 incr_state，这里简化了，没有实际的增量解码逻辑

