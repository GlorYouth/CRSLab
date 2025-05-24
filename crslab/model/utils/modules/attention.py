# -*- coding: utf-8 -*-
# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBatch(nn.Module):
    """
    针对批处理数据的自注意力机制。
    主要用于对一个批次内的多个样本的表示进行加权平均，得到整个批次的单一表示。
    例如，可以用于聚合一个批次中所有用户或物品的嵌入。
    """

    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        """
        参数:
            dim (int): 输入特征的维度。
            da (int): 注意力机制内部隐藏层的维度。
            alpha (float): LeakyReLU的负斜率，未使用在此实现中。
            dropout (float): Dropout的概率，未使用在此实现中。
        """
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha  # alpha 未在此模块的当前实现中使用
        self.dropout = dropout  # dropout 未在此模块的当前实现中使用
        # 注意力网络的参数 a 和 b
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        # 使用 Xavier均匀分布初始化参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        """
        前向传播。
        参数:
            h (torch.Tensor): 输入张量，形状为 (N, dim)，N是批次中的样本数量。
        返回:
            torch.Tensor: 加权聚合后的表示，形状为 (dim)。
        """
        # h: (N, dim)
        # 计算注意力得分
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)  # (N)
        # 应用softmax获取注意力权重
        attention = F.softmax(e, dim=0)  # (N)
        # 使用注意力权重对输入h进行加权求和
        return torch.matmul(attention, h)  # (dim)


class SelfAttentionSeq(nn.Module):
    """
    针对序列数据的自注意力机制。
    用于对序列中的每个元素计算注意力权重，并得到序列的加权表示。
    常用于文本序列、时间序列等。
    """

    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        """
        参数:
            dim (int): 输入序列中每个元素的特征维度。
            da (int): 注意力机制内部隐藏层的维度。
            alpha (float): LeakyReLU的负斜率，未使用在此实现中。
            dropout (float): Dropout的概率，未使用在此实现中。
        """
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha  # alpha 未在此模块的当前实现中使用
        self.dropout = dropout  # dropout 未在此模块的当前实现中使用
        # 注意力网络的参数 a 和 b
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        # 使用 Xavier均匀分布初始化参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        前向传播。
        参数:
            h (torch.Tensor): 输入张量，形状为 (batch, seq_len, dim)。
            mask (torch.Tensor, optional): 掩码张量，形状为 (batch, seq_len)。
                                           为True的位置表示padding，在计算注意力时不应考虑。
                                           例如: mask==[1, 1, 1, ...] 表示所有位置都是padding (这似乎与注释不符，通常mask为True表示忽略)
                                           更常见的做法是：mask中True/1表示padding，False/0表示有效内容。
            return_logits (bool, optional): 是否返回注意力权重。默认为False。
        返回:
            torch.Tensor or tuple: 如果 return_logits 为False，返回加权聚合后的序列表示，形状为 (batch, dim)。
                                   如果 return_logits 为True，返回一个元组，包含聚合表示和注意力权重 (batch, seq_len)。
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        # 计算注意力原始得分
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)

        if mask is not None:
            # 根据KGSF原作者的实现逻辑，mask中True代表padding。
            # 我们需要将padding位置的注意力得分设为一个非常小的值，使其在softmax后接近0。
            # full_mask: 将mask中为True（padding）的位置变成-1e30，False（有效）的位置变成0。
            full_mask = (mask == True).float() * -1e30

            # batch_mask: 检查序列是否完全由padding组成。
            # (mask == False) 会得到一个布尔张量，有效位为True。
            # torch.sum((mask == False), -1) 计算每行有效位的数量。
            # .bool() 转换为布尔类型，如果有效位数大于0则为True。
            # .float().unsqueeze(-1) 转换为浮点数并扩展维度，如果序列有有效内容则为1.0，否则为0.0。
            # 这样做的目的是，如果整个序列都是padding，那么full_mask将不会被应用（因为乘以0），
            # 这种情况下softmax依然会在所有-1e30上计算，可能导致NaN。
            # 更稳健的做法是确保至少有一个非padding元素，或者特殊处理全padding情况。
            # KGSF原逻辑似乎是想让全padding序列的注意力分散。
            # 此处的 batch_mask 逻辑似乎有点复杂，通常直接应用 full_mask 即可。
            # 为了保持与原意图一致（尽管可能存在改进空间），我们暂时保留类似结构，但简化处理。
            # 如果一个序列全是padding, 那么它的所有注意力得分都会被设为-1e30。
            # 如果一个序列有非padding内容，则只对padding位置的得分应用-1e30。
            e = e + full_mask.unsqueeze(-1)  # 直接应用掩码

        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1) - 计算注意力权重

        # 使用注意力权重对输入h的序列表示进行加权求和
        # torch.transpose(attention, 1, 2) -> (batch, 1, seq_len)
        # torch.matmul(torch.transpose(attention, 1, 2), h) -> (batch, 1, dim)
        # .squeeze(1) -> (batch, dim)
        aggregated_h = torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)

        if return_logits:
            return aggregated_h, attention.squeeze(-1)  # 返回聚合后的表示和注意力权重
        else:
            return aggregated_h  # 只返回聚合后的表示


class SelfAttentionNetwork(nn.Module):
    """
    基于 nn.MultiheadAttention 的自注意力模块封装。
    """

    def __init__(self, input_size, hidden_size, output_size, num_heads=4, dropout=0.1):
        super(SelfAttentionNetwork, self).__init__()

        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, attention_mask=None):
        # inputs: (N, L, E) - 输入张量，N为批量大小，L为序列长度，E为特征维度
        # attention_mask: (N, L) - 注意力掩码，N为批量大小，L为序列长度
        # nn.MultiheadAttention期望的输入是(L, N, E)，所以需要转置
        query = self.query_linear(inputs).transpose(0, 1)  # (L, N, H) - 查询向量，H为隐藏层维度
        key = self.key_linear(inputs).transpose(0, 1)  # (L, N, H) - 键向量
        value = self.value_linear(inputs).transpose(0, 1)  # (L, N, H) - 值向量

        # multihead_attn 的 attention_mask (key_padding_mask) 应该是 (N, L)
        # 其中 True 表示一个被掩码（填充）的位置，这些位置在计算注意力时将被忽略。
        # False 表示一个未被掩码的位置。

        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=attention_mask)  # (L, N, H) - 注意力输出
        attn_output = attn_output.transpose(0, 1)  # (N, L, H) - 转置回 (N, L, H)

        output = self.output_linear(attn_output)  # (N, L, O) - 最终输出，O为输出维度
        return output


class ContextFusionGate(nn.Module):
    """
    门控机制，用于融合词级别和实体级别的上下文表示。
    """

    def __init__(self, word_context_dim, entity_context_dim, decoder_state_dim, output_dim):
        """
        参数:
            word_context_dim (int): 词级别上下文的维度。
            entity_context_dim (int): 实体级别上下文的维度。
            decoder_state_dim (int): 解码器查询状态的维度。
            output_dim (int): 融合后输出上下文的维度。
                              假设如果 word_context 和 entity_context 的维度不同，它们将被投影到此维度。
        """
        super(ContextFusionGate, self).__init__()
        self.word_context_dim = word_context_dim
        self.entity_context_dim = entity_context_dim
        self.decoder_state_dim = decoder_state_dim
        self.output_dim = output_dim

        # 如果需要，使用线性层将上下文投影到相同的维度
        if self.word_context_dim != self.output_dim:
            self.word_proj = nn.Linear(self.word_context_dim, self.output_dim)
        else:
            self.word_proj = nn.Identity()  # 如果维度相同，则使用 Identity 层

        if self.entity_context_dim != self.output_dim:
            self.entity_proj = nn.Linear(self.entity_context_dim, self.output_dim)
        else:
            self.entity_proj = nn.Identity()  # 如果维度相同，则使用 Identity 层

        # 用于计算门控值的线性层
        # 该层的输入将是以下各项的拼接：
        #   - 投影后的词上下文
        #   - 投影后的实体上下文
        #   - 解码器状态
        # 输出维度为 output_dim，这个门控值将用于逐元素调节词上下文的权重。
        self.gate_linear = nn.Linear(self.output_dim + self.output_dim + self.decoder_state_dim, self.output_dim)

    def forward(self, word_context, entity_context, decoder_state):
        """
        根据解码器状态融合词上下文和实体上下文。

        参数:
            word_context (torch.Tensor): 词级别上下文表示。
                                         形状: (batch_size, seq_len_word, word_context_dim) 或 (batch_size, word_context_dim)
                                         为简单起见，如果输入是3D的，本方法内部会先进行平均池化处理为2D。
                                         在实际应用中，可能需要更复杂的序列编码方式。
            entity_context (torch.Tensor): 实体级别上下文表示。
                                           形状: (batch_size, seq_len_entity, entity_context_dim) 或 (batch_size, entity_context_dim)
                                           同上，如果输入是3D的，会进行平均池化。
            decoder_state (torch.Tensor): 解码器的当前查询状态 (例如，Transformer解码器某层的输出或特定的查询向量)。
                                          形状: (batch_size, decoder_state_dim)

        返回:
            torch.Tensor: 融合后的上下文表示。形状: (batch_size, output_dim)
        """
        # 确保上下文输入是二维的 (batch_size, dim)
        # 这种处理方式比较简单，实际应用中可能需要更精细的序列表示方法，
        # 例如，使用 SelfAttentionSeq 对序列本身进行注意力加权，得到固定长度的向量。
        if word_context.ndim > 2:
            word_context = torch.mean(word_context, dim=1)
        if entity_context.ndim > 2:
            entity_context = torch.mean(entity_context, dim=1)

        # 将上下文投影到 output_dim
        proj_word_context = self.word_proj(word_context)  # (batch_size, output_dim)
        proj_entity_context = self.entity_proj(entity_context)  # (batch_size, output_dim)

        # 拼接用于门控计算的输入
        gate_input = torch.cat([proj_word_context, proj_entity_context, decoder_state], dim=-1)

        # 计算门控值 (用于 word_context，entity_context 的门控值将是 1-gate_value)
        # gate_value 决定了词级别上下文的权重。
        gate_value = torch.sigmoid(self.gate_linear(gate_input))  # (batch_size, output_dim)

        # 使用门控值融合上下文
        # 逐元素相乘实现门控
        fused_context = gate_value * proj_word_context + (1 - gate_value) * proj_entity_context

        return fused_context


class GatedLinearUnit(nn.Module):
    """
    门控线性单元 (Gated Linear Unit, GLU)。
    """

    def __init__(self, input_size, output_size=None, bias=True):
        """
        参数:
            input_size (int): 输入特征的维度。
            output_size (int, optional): 输出特征的维度。如果为None，则默认为 input_size // 2。
            bias (bool): 线性层是否使用偏置。
        """
        super(GatedLinearUnit, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size else input_size // 2
        self.bias = bias

        # 线性层将输入映射到两倍的输出维度，一半用于值，一半用于门控
        self.linear = nn.Linear(self.input_size, self.output_size * 2, bias=self.bias)

    def forward(self, inputs):
        """
        前向传播。
        参数:
            inputs (torch.Tensor): 输入张量，形状为 (..., input_size)。
        返回:
            torch.Tensor: GLU的输出，形状为 (..., output_size)。
        """
        outputs = self.linear(inputs)  # (..., output_size * 2)
        value = outputs[..., :self.output_size]  # 取前一半作为值
        gate = torch.sigmoid(outputs[..., self.output_size:])  # 取后一半并通过sigmoid作为门控
        return value * gate  # 逐元素相乘


if __name__ == '__main__':
    # SelfAttentionBatch 使用示例
    print("\n--- SelfAttentionBatch 示例 ---")
    dim_batch, da_batch = 32, 16
    sab = SelfAttentionBatch(dim=dim_batch, da=da_batch)
    h_batch = torch.randn(5, dim_batch)  # 5个样本，每个样本32维
    output_sab = sab(h_batch)
    print("SelfAttentionBatch 输入形状:", h_batch.shape)
    print("SelfAttentionBatch 输出形状:", output_sab.shape)  # 预期: (32)

    # SelfAttentionSeq 使用示例
    print("\n--- SelfAttentionSeq 示例 ---")
    batch_size_seq, seq_len_seq, dim_seq, da_seq = 4, 10, 32, 16
    sas = SelfAttentionSeq(dim=dim_seq, da=da_seq)
    h_seq = torch.randn(batch_size_seq, seq_len_seq, dim_seq)
    # 示例掩码：第一个样本的前5个token有效，第二个样本的前7个token有效，其余为padding
    mask_seq = torch.ones(batch_size_seq, seq_len_seq, dtype=torch.bool)  # True表示padding
    mask_seq[0, :5] = False
    mask_seq[1, :7] = False
    mask_seq[2, :] = True  # 第三个样本全是padding
    mask_seq[3, :seq_len_seq] = False  # 第四个样本全有效

    output_sas = sas(h_seq, mask=mask_seq)
    output_sas_logits, logits_sas = sas(h_seq, mask=mask_seq, return_logits=True)
    print("SelfAttentionSeq 输入形状 (h):", h_seq.shape)
    print("SelfAttentionSeq 输入形状 (mask):", mask_seq.shape)
    print("SelfAttentionSeq 输出形状:", output_sas.shape)  # 预期: (4, 32)
    print("SelfAttentionSeq 输出形状 (带logits):", output_sas_logits.shape)  # 预期: (4, 32)
    print("SelfAttentionSeq 注意力权重形状:", logits_sas.shape)  # 预期: (4, 10)
    print("示例注意力权重 (第一个样本):", logits_sas[0])
    print("示例注意力权重 (全padding样本):", logits_sas[2])  # 应该接近均匀分布或集中在某个位置，取决于实现细节

    # SelfAttentionNetwork 使用示例
    print("\n--- SelfAttentionNetwork 示例 ---")
    batch_size_san, seq_len_san, embed_dim_san = 4, 10, 32
    hidden_dim_san, output_dim_san = 64, 32

    self_attn_net = SelfAttentionNetwork(embed_dim_san, hidden_dim_san, output_dim_san)

    test_input_san = torch.rand(batch_size_san, seq_len_san, embed_dim_san)
    # 创建一个虚拟的注意力掩码 (例如，第一个批处理元素的最后2个项目是填充的)
    test_mask_san = torch.zeros(batch_size_san, seq_len_san, dtype=torch.bool)  # False表示有效，True表示padding
    test_mask_san[0, -2:] = True

    output_san = self_attn_net(test_input_san, attention_mask=test_mask_san)
    print("SelfAttentionNetwork 输入形状:", test_input_san.shape)
    print("SelfAttentionNetwork Mask形状:", test_mask_san.shape)
    print("SelfAttentionNetwork 输出形状:", output_san.shape)  # 预期: (4, 10, 32)

    # ContextFusionGate 使用示例
    print("\n--- ContextFusionGate 示例 ---")
    bs_cfg = 4  # 批量大小
    wc_dim_cfg, ec_dim_cfg, ds_dim_cfg, o_dim_cfg = 100, 120, 80, 90  # 示例维度

    # 情况1：上下文已经是二维的 (batch_size, dim)
    word_ctx_2d_cfg = torch.randn(bs_cfg, wc_dim_cfg)
    entity_ctx_2d_cfg = torch.randn(bs_cfg, ec_dim_cfg)
    decoder_s_cfg = torch.randn(bs_cfg, ds_dim_cfg)

    fusion_gate = ContextFusionGate(wc_dim_cfg, ec_dim_cfg, ds_dim_cfg, o_dim_cfg)
    fused_output_2d_cfg = fusion_gate(word_ctx_2d_cfg, entity_ctx_2d_cfg, decoder_s_cfg)
    print("ContextFusionGate 输出形状 (2D 输入):", fused_output_2d_cfg.shape)  # 预期: (4, 90)

    # 情况2：上下文是三维的 (batch_size, seq_len, dim)
    seq_len_w_cfg, seq_len_e_cfg = 5, 3
    word_ctx_3d_cfg = torch.randn(bs_cfg, seq_len_w_cfg, wc_dim_cfg)
    entity_ctx_3d_cfg = torch.randn(bs_cfg, seq_len_e_cfg, ec_dim_cfg)

    fused_output_3d_cfg = fusion_gate(word_ctx_3d_cfg, entity_ctx_3d_cfg, decoder_s_cfg)
    print("ContextFusionGate 输出形状 (3D 输入，使用平均池化):", fused_output_3d_cfg.shape)  # 预期: (4, 90)

    # GatedLinearUnit 使用示例
    print("\n--- GatedLinearUnit 示例 ---")
    glu = GatedLinearUnit(input_size=128)
    test_input_glu = torch.randn(bs_cfg, 128)
    output_glu = glu(test_input_glu)
    print("GatedLinearUnit 输出形状:", output_glu.shape)  # 预期: (4, 64)

    glu_custom_out = GatedLinearUnit(input_size=128, output_size=30)
    output_glu_custom = glu_custom_out(test_input_glu)
    print("GatedLinearUnit 自定义输出形状:", output_glu_custom.shape)  # 预期: (4, 30)
