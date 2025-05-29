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
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        # h: (N, dim)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)  # (N)
        return torch.matmul(attention, h)  # (dim)


class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)  # for all padding one, the mask=0
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)


class SelfAttentionSeqImproved(nn.Module):
    def __init__(self, dim, da, num_heads=1, dropout_rate=0.1, use_bias=True, activation_fn='tanh'):
        """
        自注意力模块的初始化函数。

        参数:
            dim (int): 输入特征的维度。
            da (int): 每个注意力头内部中间层的维度。
            num_heads (int): 注意力头的数量。默认为1（单头注意力）。
            dropout_rate (float): Dropout的比率。默认为0.1。
            use_bias (bool): 线性层是否使用偏置项。默认为True。
            activation_fn (str): 内部使用的激活函数名称 ('tanh', 'relu', 'gelu')。默认为'tanh'。
        """
        super(SelfAttentionSeqImproved, self).__init__()
        assert dim % num_heads == 0, "输入维度 'dim' 必须能够被注意力头数量 'num_heads' 整除"

        self.dim = dim  # 输入总维度
        self.da = da    # 每个头内部的注意力计算维度
        self.num_heads = num_heads # 注意力头的数量
        self.head_dim = dim // num_heads # 每个注意力头的输出维度
        self.dropout_rate = dropout_rate

        # 用于注意力打分机制的线性层:
        # 原始思路是 h -> tanh(h @ W1 + b1) @ W2 + b2
        # 这里 W1 将输入 h 投影到 da * num_heads 维
        # W2 将 da * num_heads 维投影到 num_heads 维 (每个头一个分数)
        self.fc1 = nn.Linear(self.dim, self.da * self.num_heads, bias=use_bias)
        self.fc2 = nn.Linear(self.da * self.num_heads, self.num_heads, bias=use_bias) # 输出每个头的注意力原始分数

        # 选择激活函数
        if activation_fn == 'tanh':
            self.activation = torch.tanh
        elif activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {activation_fn}")

        self.dropout = nn.Dropout(dropout_rate) # Dropout层

        # 如果是多头注意力，并且最终需要合并回原始维度，则通常会有一个输出线性层
        if self.num_heads > 1:
            # 这个线性层将拼接后的多头输出 (dim) 重新投影到 dim 维
            # 在这个特定实现中，由于我们目标是输出 (batch, dim)，这个fc_out是合理的
            self.fc_out = nn.Linear(dim, dim, bias=use_bias)
        else:
            self.fc_out = None # 单头注意力通常不需要额外的输出投影层来合并

        self._init_weights() # 初始化权重

    def _init_weights(self):
        # 初始化权重，有助于模型训练
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        if self.fc_out is not None and hasattr(self.fc_out, 'weight'): # 确保fc_out存在且是nn.Linear
            nn.init.xavier_uniform_(self.fc_out.weight)
            if self.fc_out.bias is not None:
                nn.init.zeros_(self.fc_out.bias)

    def forward(self, h, mask=None, return_logits=False):
        """
        前向传播函数。

        参数:
            h (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, dim)。
            mask (torch.Tensor, optional): 掩码张量，形状为 (batch_size, seq_len)。
                                           为True的位置表示是padding token，应该被忽略。默认为None。
            return_logits (bool): 是否返回注意力权重（softmax后的值）。默认为False。

        返回:
            torch.Tensor: 上下文向量，形状为 (batch_size, dim)。
            torch.Tensor (optional): 如果 return_logits 为 True，则额外返回注意力权重，
                                     形状为 (batch_size, num_heads, seq_len) 或 (batch_size, seq_len)。
        """
        batch_size, seq_len, _ = h.shape # 获取输入的形状

        # 1. 计算注意力分数
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, da * num_heads)
        h_projected = self.activation(self.fc1(h)) # 通过第一个线性层和激活函数

        # (batch_size, seq_len, da * num_heads) -> (batch_size, seq_len, num_heads)
        e = self.fc2(h_projected) # 通过第二个线性层得到每个头的原始注意力分数

        # 为了方便后续在每个头内部独立进行Softmax，调整e的形状
        # (batch_size, seq_len, num_heads) -> (batch_size, num_heads, seq_len)
        e = e.transpose(1, 2)

        # 2. 应用掩码 (Masking)
        if mask is not None:
            # mask: (batch_size, seq_len)
            # extended_mask: (batch_size, 1, seq_len) 以便广播到 e 的形状 (batch_size, num_heads, seq_len)
            extended_mask = mask.unsqueeze(1)
            # 将掩码为True（即padding token）位置的注意力分数设置为一个非常小的负数
            # 这样在Softmax之后，这些位置的权重会趋近于0
            mask_value = torch.finfo(e.dtype).min  # 获取 e 数据类型的最小值
            e = e.masked_fill(extended_mask, mask_value)

        # 3. 计算注意力权重 (Attention Weights)
        # 沿序列长度维度 (dim=-1) 进行Softmax，得到归一化的注意力权重
        # 形状: (batch_size, num_heads, seq_len)
        attention_weights = F.softmax(e, dim=-1)
        attention_weights = self.dropout(attention_weights) # 对注意力权重应用Dropout

        # 4. 计算上下文向量 (Weighted Sum)
        # 为了进行多头注意力的加权求和，需要将输入h也调整为适应多头的形式
        # h: (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads, head_dim)
        h_reshaped = h.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 调整h_reshaped的维度顺序以匹配批量的矩阵乘法 (BMM)
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        h_reshaped_permuted = h_reshaped.permute(0, 2, 1, 3)

        # 计算加权和
        # attention_weights: (batch_size, num_heads, seq_len)
        # 为了进行BMM，将其扩展一维: (batch_size, num_heads, 1, seq_len)
        # h_reshaped_permuted: (batch_size, num_heads, seq_len, head_dim)
        # matmul结果: (batch_size, num_heads, 1, head_dim)
        # squeeze后: (batch_size, num_heads, head_dim)
        # 这表示每个头都对序列信息进行了加权汇总
        context_per_head = torch.matmul(attention_weights.unsqueeze(2), h_reshaped_permuted).squeeze(2)

        # 5. 合并多头输出 (如果num_heads > 1)
        if self.num_heads > 1:
            # 将多个头的输出拼接起来
            # (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            # num_heads * head_dim == dim
            # 形状变为: (batch_size, dim)
            context = context_per_head.contiguous().view(batch_size, self.dim)
            if self.fc_out is not None:
                context = self.fc_out(context) # 通过最后的线性层进行输出变换
        else:
            # 单头情况下，直接移除多余的num_heads维度
            # (batch_size, 1, head_dim) -> (batch_size, head_dim)
            # head_dim == dim
            context = context_per_head.squeeze(1)

        if return_logits:
            # 返回上下文向量和注意力权重
            # 如果是多头，可以返回每个头的权重，或者平均/拼接后的权重
            # 这里简单返回原始的、经过Softmax和Dropout的注意力权重
            # (batch_size, num_heads, seq_len)
            return context, attention_weights
        else:
            return context
