# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/29, 2021/1/4
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
KGSF
====
参考文献:
    Zhou, Kun, et al. `"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`_ KDD 2020.

.. _`"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`:
   https://dl.acm.org/doi/abs/10.1145/3394486.3403143

"""

import os  # 操作系统接口模块

import numpy as np  # 科学计算库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch神经网络函数库
from loguru import logger  # 日志记录库
from torch import nn  # PyTorch神经网络模块
from torch_geometric.nn import GCNConv, RGCNConv  # PyTorch Geometric图神经网络卷积层

from crslab.config import MODEL_PATH  # 模型路径配置
from crslab.model.base import BaseModel  # 模型基类
from crslab.model.utils.functions import edge_to_pyg_format  # 边列表转PyG格式工具函数
from crslab.model.utils.modules.attention import SelfAttentionSeq  # 自注意力序列模块
from crslab.model.utils.modules.transformer import TransformerEncoder  # Transformer编码器模块
from .modules import GateLayer, TransformerDecoderKG  # 当前目录下的门控层和带知识图谱的Transformer解码器模块
from .resources import resources  # 当前目录下的资源


class KGSFModel(BaseModel):  # KGSF模型类，继承自BaseModel
    """

    属性:
        vocab_size: 整数，表示词汇表大小。
        pad_token_idx: 整数，表示填充标记的ID。
        start_token_idx: 整数，表示开始标记的ID。
        end_token_idx: 整数，表示结束标记的ID。
        token_emb_dim: 整数，表示标记嵌入层的维度。
        pretrain_embedding: 字符串，表示预训练嵌入文件的路径。
        n_word: 整数，表示单词数量。
        n_entity: 整数，表示实体数量。
        pad_word_idx: 整数，表示单词填充的ID。
        pad_entity_idx: 整数，表示实体填充的ID。
        num_bases: 整数，表示RGCN中的基数量。
        kg_emb_dim: 整数，表示知识图谱嵌入的维度。
        n_heads: 整数，表示Transformer中的头数量。
        n_layers: 整数，表示Transformer中的层数量。
        ffn_size: 整数，表示前馈网络隐藏层的大小。
        dropout: 浮点数，表示dropout比率。
        attention_dropout: 整数，表示注意力层的dropout比率。
        relu_dropout: 整数，表示ReLU层的dropout比率。
        learn_positional_embeddings: 布尔值，表示是否学习位置嵌入。
        embeddings_scale: 布尔值，表示是否使用嵌入缩放。
        reduction: 布尔值，表示是否使用归约。
        n_positions: 整数，表示最大位置数量。
        response_truncate: 整数，表示回复生成的最大长度。
        pretrained_embedding: 字符串，表示预训练词嵌入的路径。

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        参数:
            opt (dict): 包含超参数的字典。
            device (torch.device): 指定数据和模型存放的设备。
            vocab (dict): 包含词汇表信息的字典。
            side_data (dict): 包含辅助数据的字典。

        """
        self.device = device  # 设备
        self.gpu = opt.get("gpu", [-1])  # 获取GPU配置，默认为-1 (CPU)
        # 词汇表相关参数
        self.vocab_size = vocab['vocab_size']  # 词汇表大小
        self.pad_token_idx = vocab['pad']  # 填充标记ID
        self.start_token_idx = vocab['start']  # 开始标记ID
        self.end_token_idx = vocab['end']  # 结束标记ID
        self.token_emb_dim = opt['token_emb_dim']  # Token嵌入维度
        self.pretrained_embedding = side_data.get('embedding', None)  # 预训练词嵌入路径
        # 知识图谱相关参数
        self.n_word = vocab['n_word']  # 词汇表中词的数量（用于概念图）
        self.n_entity = vocab['n_entity']  # 实体数量（用于实体图）
        self.pad_word_idx = vocab['pad_word']  # 词填充ID
        self.pad_entity_idx = vocab['pad_entity']  # 实体填充ID
        entity_kg = side_data['entity_kg']  # 实体知识图谱数据
        self.n_relation = entity_kg['n_relation']  # 实体图中的关系数量
        entity_edges = entity_kg['edge']  # 实体图的边
        # 将实体图的边转换为PyTorch Geometric格式的索引和类型
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)  # 实体图边索引移至设备
        self.entity_edge_type = self.entity_edge_type.to(device)  # 实体图边类型移至设备
        word_edges = side_data['word_kg']['edge']  # 概念知识图谱（词图）的边

        # 将概念图的边转换为PyTorch Geometric格式并移至设备
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)

        self.num_bases = opt['num_bases']  # RGCN的基分解数量
        self.kg_emb_dim = opt['kg_emb_dim']  # 知识图谱嵌入维度
        # Transformer相关参数
        self.n_heads = opt['n_heads']  # 多头注意力头数
        self.n_layers = opt['n_layers']  # Transformer层数
        self.ffn_size = opt['ffn_size']  # 前馈网络隐藏层大小
        self.dropout = opt['dropout']  # Dropout概率
        self.attention_dropout = opt['attention_dropout']  # 注意力Dropout概率
        self.relu_dropout = opt['relu_dropout']  # ReLU激活函数后的Dropout概率
        self.learn_positional_embeddings = opt['learn_positional_embeddings']  # 是否学习位置嵌入
        self.embeddings_scale = opt['embeddings_scale']  # 是否缩放嵌入
        self.reduction = opt['reduction']  # Transformer编码器输出是否进行归约
        self.n_positions = opt['n_positions']  # 最大序列长度（位置编码）
        self.response_truncate = opt.get('response_truncate', 20)  # 回复截断长度，默认为20
        # 复制机制相关
        dataset = opt['dataset']  # 数据集名称
        dpath = os.path.join(MODEL_PATH, "kgsf", dataset)  # 模型特定数据集的路径
        resource = resources[dataset]  # 获取数据集对应的资源信息
        super(KGSFModel, self).__init__(opt, device, dpath, resource)  # 调用父类初始化方法

    def build_model(self):  # 构建模型各组件
        self._init_embeddings()  # 初始化嵌入层
        self._build_kg_layer()  # 构建知识图谱层
        self._build_infomax_layer()  # 构建互信息最大化层 (用于预训练)
        self._build_recommendation_layer()  # 构建推荐层
        self._build_conversation_layer()  # 构建对话层

    def _init_embeddings(self):  # 初始化嵌入层
        if self.pretrained_embedding is not None:  # 如果提供了预训练词嵌入
            self.token_embedding = nn.Embedding.from_pretrained(  # 从预训练权重加载
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,  # 不冻结权重
                padding_idx=self.pad_token_idx)
        else:  # 否则，随机初始化
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)  # 正态分布初始化
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)  # 填充部分权重设为0

        # 初始化概念图（词图）的嵌入层
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)  # 正态分布初始化
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)  # 填充部分权重设为0

        logger.debug('[Finish init embeddings]')  # 日志记录：完成嵌入层初始化

    def _build_kg_layer(self):  # 构建知识图谱相关层
        # 实体图编码器 (DB encoder)
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)  # RGCN层
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 实体自注意力层

        # 概念图编码器 (Concept encoder)
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)  # GCN层
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 词自注意力层

        # 门控机制，用于融合实体和概念信息
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logger.debug('[Finish build kg layer]')  # 日志记录：完成知识图谱层构建

    def _build_infomax_layer(self):  # 构建互信息最大化层 (用于知识图谱预训练)
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)  # 线性变换层
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 线性偏置层
        self.infomax_loss = nn.MSELoss(reduction='sum')  # 均方误差损失函数

        logger.debug('[Finish build infomax layer]')  # 日志记录：完成互信息最大化层构建

    def _build_recommendation_layer(self):  # 构建推荐层
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 推荐偏置线性层 (输出对每个实体的评分)
        self.rec_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数 (用于推荐任务)

        logger.debug('[Finish build rec layer]')  # 日志记录：完成推荐层构建

    def _build_conversation_layer(self):  # 构建对话生成层
        # 注册一个名为'START'的缓冲区，存储开始标记的张量
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        # 对话编码器 (Transformer Encoder)
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.token_embedding,  # 使用共享的词嵌入层
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.reduction,
            n_positions=self.n_positions,
        )

        # 用于将知识图谱信息融入对话的线性变换层
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 实体表示变换
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 实体注意力表示变换
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 词表示变换
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 词注意力表示变换

        # 复制机制相关层
        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)  # 融合解码器隐状态、实体上下文、概念上下文
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)  # 输出复制概率分布
        # 加载预先计算好的复制掩码 (哪些词可以被复制)
        self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                         ).to(self.device)

        # 对话解码器 (Transformer Decoder with KG)
        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,  # 使用共享的词嵌入层
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)  # 对话生成的损失函数 (忽略填充标记)

        logger.debug('[Finish build conv layer]')  # 日志记录：完成对话层构建

    def pretrain_infomax(self, batch):  # 知识图谱预训练 (互信息最大化)
        """
        此方法通过最大化词图表示和实体图表示之间的互信息来预训练知识图谱嵌入。
        words: (batch_size, word_length) 输入的词序列
        entity_labels: (batch_size, n_entity) 实体标签 (one-hot或multi-hot)
        """
        words, entity_labels = batch  # 解包输入

        loss_mask = torch.sum(entity_labels)  # 计算有效的标签数量，用于归一化损失
        if loss_mask.item() == 0:  # 如果没有有效标签，则不计算损失
            return None

        # 获取实体图和词图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 从词图中获取输入词序列的表示
        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len) 生成词序列的填充掩码

        # 对词表示应用自注意力机制
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim) 互信息表示，经过线性变换
        # 预测实体 (点积相似度后加偏置)
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        loss = self.infomax_loss(info_predict, entity_labels) / loss_mask  # 计算MSE损失并归一化
        return loss

    def recommend(self, batch, mode):  # 推荐模块
        """
        生成推荐结果。
        context_entities: (batch_size, entity_length) 上下文实体序列
        context_words: (batch_size, word_length) 上下文词序列
        entities: (batch_size, n_entity) 当前对话轮次相关的实体，用于计算辅助的infomax loss
        movie: (batch_size) 目标推荐的电影/项目 ID
        """
        context_entities, context_words, entities, movie = batch  # 解包输入

        # 获取实体图和词图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 生成上下文实体和词的填充掩码
        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        # 获取上下文实体和词的表示
        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        # 对上下文实体和词的表示应用自注意力机制
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        # 使用门控机制融合实体和词的注意力表示，得到用户表示
        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        # 计算推荐得分 (用户表示与所有实体表示的点积 + 偏置)
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, movie)  # 计算推荐任务的交叉熵损失

        # 计算辅助的互信息损失 (与预训练阶段类似，但使用当前对话上下文的词信息预测相关实体)
        info_loss_mask = torch.sum(entities)
        if info_loss_mask.item() == 0:  # 如果没有相关实体，则不计算此损失
            info_loss = None
        else:
            word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (bs, #entity)
            info_loss = self.infomax_loss(info_predict, entities) / info_loss_mask

        return rec_loss, info_loss, rec_scores  # 返回推荐损失、互信息损失和推荐得分

    def freeze_parameters(self):  # 冻结指定模块的参数 (通常在微调或特定训练阶段使用)
        # 定义需要冻结参数的模块列表
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.infomax_norm, self.rec_bias]
        for model in freeze_models:  # 遍历模块列表
            for p in model.parameters():  # 遍历模块中的每个参数
                p.requires_grad = False  # 设置参数不需要梯度，即冻结

    def _starts(self, batch_size):  # 生成解码器初始输入 (开始标记)
        """返回 bsz 个开始标记."""
        return self.START.detach().expand(batch_size, 1)  # 将单个开始标记扩展到batch_size

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):  # 使用Teacher Forcing的解码（结合知识图谱）
        """
        在训练阶段使用Teacher Forcing进行解码。
        token_encoding: 对话历史的Transformer编码器输出 ((encoder_output, encoder_hidden), (encoder_output_mask))
        entity_reps: 上下文实体表示 (bs, entity_len, kg_dim)
        entity_emb_attn: 实体上下文的注意力加权表示 (bs, kg_dim)
        entity_mask: 实体序列的填充掩码 (bs, entity_len)
        word_reps: 上下文词表示 (bs, word_len, kg_dim)
        word_emb_attn: 词上下文的注意力加权表示 (bs, kg_dim)
        word_mask: 词序列的填充掩码 (bs, word_len)
        response: 目标回复序列 (bs, seq_len)
        """
        batch_size, seq_len = response.shape  # 获取批次大小和序列长度
        start_tokens = self._starts(batch_size)  # 获取开始标记
        # 解码器输入：将开始标记与目标回复（除去最后一个token）拼接
        inputs = torch.cat((start_tokens, response[:, :-1]), dim=-1).long()

        # 使用TransformerDecoderKG进行解码
        # dialog_latent: 解码器在每个时间步的输出隐状态
        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)
        # 将实体和词的上下文表示扩展到与解码序列长度一致
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, ffn_size)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, ffn_size)

        # 拼接解码器隐状态、实体上下文、概念上下文，并通过线性层进行融合，用于复制机制
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, token_emb_dim)

        # 计算复制概率和生成概率
        # copy_logits: 从上下文中复制词的logits
        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
            0)  # (bs, seq_len, vocab_size)，乘以copy_mask确保只复制允许的词
        # gen_logits: 从词汇表生成词的logits
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        sum_logits = copy_logits + gen_logits  # 最终的logits是复制和生成的logits之和
        preds = sum_logits.argmax(dim=-1)  # 预测的词索引
        return sum_logits, preds  # 返回logits和预测结果

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):  # 贪婪解码（结合知识图谱）
        """
        在推理阶段使用贪婪搜索进行解码。
        参数与 _decode_forced_with_kg 类似，但不包含 response。
        """
        batch_size = token_encoding[0].shape[0]  # 获取批次大小
        inputs = self._starts(batch_size).long()  # 解码器初始输入为开始标记
        incr_state = None  # 用于存储解码器自回归的中间状态
        logits_list = []  # 存储每个时间步的logits
        for _ in range(self.response_truncate):  # 循环生成直到达到最大长度
            # 单步解码
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim) 只取最后一个时间步的输出
            # 准备复制机制的上下文输入
            db_latent = entity_emb_attn.unsqueeze(1)  # (bs, 1, ffn_size)
            concept_latent = word_emb_attn.unsqueeze(1)  # (bs, 1, ffn_size)
            copy_latent = self.copy_norm(
                torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))  # (bs, 1, token_emb_dim)

            # 计算复制和生成logits
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
                0)  # (bs, 1, vocab_size)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, 1, vocab_size)
            sum_logits = copy_logits + gen_logits  # (bs, 1, vocab_size)
            preds = sum_logits.argmax(dim=-1).long()  # (bs, 1) 贪婪选择概率最大的词
            logits_list.append(sum_logits)  # 保存当前时间步的logits
            inputs = torch.cat((inputs, preds), dim=1)  # 将预测的词加入到输入序列中，用于下一步解码

            # 检查是否所有序列都已生成结束标记
            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:  # 如果所有序列都结束，则停止生成
                break
        logits = torch.cat(logits_list, dim=1)  # (bs, generated_seq_len, vocab_size) 拼接所有时间步的logits
        return logits, inputs  # 返回完整logits和生成的序列

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):  # Beam Search解码（结合知识图谱）
        """
        在推理阶段使用Beam Search进行解码。
        beam: Beam Search的宽度。
        其他参数与 _decode_greedy_with_kg 类似。
        """
        batch_size = token_encoding[0].shape[0]
        # 初始化输入，(1, batch_size, 1)，1表示当前候选序列数量
        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None  # 解码器增量状态

        # sequences 存储每个样本的beam个候选序列: [[[token_ids_list], [logits_list], probability_score]]
        sequences = [[[list(), list(), 1.0]]] * batch_size  # 初始化，每个样本一个空序列，概率为1.0

        for i in range(self.response_truncate):  # 遍历解码步数，最大为截断长度
            if i == 1:  # 从第二步开始，需要将上下文信息复制beam份，因为现在有beam个候选序列
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),  # (beam*bs, enc_len, dim)
                                  token_encoding[1].repeat(beam, 1))  # (beam*bs, enc_len)
                entity_reps = entity_reps.repeat(beam, 1, 1)  # (beam*bs, entity_len, kg_dim)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)  # (beam*bs, kg_dim)
                entity_mask = entity_mask.repeat(beam, 1)  # (beam*bs, entity_len)
                word_reps = word_reps.repeat(beam, 1, 1)  # (beam*bs, word_len, kg_dim)
                word_emb_attn = word_emb_attn.repeat(beam, 1)  # (beam*bs, kg_dim)
                word_mask = word_mask.repeat(beam, 1)  # (beam*bs, word_len)

            current_inputs_list = []
            num_candidates_per_sample = len(sequences[0])  # 当前每个样本的候选序列数量 (第一步是1，之后是beam)

            if i != 0:  # 非第一步解码
                # 收集所有候选序列的当前输入
                for k in range(num_candidates_per_sample):  # 遍历当前候选
                    for j in range(batch_size):  # 遍历batch中的每个样本
                        # sequences[j][k][0] 是第j个样本的第k个候选序列的token ids
                        current_inputs_list.append(sequences[j][k][0])  # 添加token id序列
                # 将列表转换为tensor: (num_candidates * batch_size, current_seq_len)
                inputs = torch.stack(current_inputs_list).reshape(num_candidates_per_sample * batch_size, -1)
            else:  # 第一步解码，输入是起始符
                inputs = inputs.reshape(batch_size, -1)  # (bs, 1)

            with torch.no_grad():  # 推理过程不计算梯度
                # 解码一步 (num_candidates_per_sample * batch_size, current_seq_len, dim)
                dialog_latent, incr_state_new = self.conv_decoder(
                    inputs,  # (current_candidates * bs, current_seq_len)
                    token_encoding, word_reps, word_mask,
                    entity_reps, entity_mask, incr_state  # incr_state可能需要根据num_candidates调整
                )
                incr_state = incr_state_new  # 更新解码器状态
                dialog_latent = dialog_latent[:, -1:, :]  # (current_candidates * bs, 1, dim)

                # 准备复制机制的上下文输入
                # 注意：entity_emb_attn等此时已经是 (beam*bs, kg_dim) 或 (bs, kg_dim) for i=0
                # 如果 i=0, entity_emb_attn 是 (bs, kg_dim)，需要扩展
                current_bs_for_kg = entity_emb_attn.shape[0]
                db_latent = entity_emb_attn.unsqueeze(1)  # (current_bs_for_kg, 1, ffn_size)
                concept_latent = word_emb_attn.unsqueeze(1)  # (current_bs_for_kg, 1, ffn_size)

                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
                sum_logits = copy_logits + gen_logits  # (current_candidates * bs, 1, vocab_size)

            # sum_logits: (num_candidates_per_sample * batch_size, 1, vocab_size)
            # Reshape to (num_candidates_per_sample, batch_size, 1, vocab_size)
            reshaped_logits = sum_logits.reshape(num_candidates_per_sample, batch_size, 1, -1)
            # 计算概率并取top-k
            probs, preds = torch.nn.functional.softmax(reshaped_logits, dim=-1).topk(beam, dim=-1)
            # probs, preds 的形状: (num_candidates_per_sample, batch_size, 1, beam)

            new_sequences_all_samples = []
            for j in range(batch_size):  # 遍历每个样本
                all_candidates_for_sample_j = []
                for k in range(num_candidates_per_sample):  # 遍历当前样本的每个旧候选序列
                    prev_seq_tokens = sequences[j][k][0]  # list of token_ids
                    prev_seq_logits = sequences[j][k][1]  # list of logits tensors
                    prev_seq_prob = sequences[j][k][2]  # float probability

                    # 从PyTorch tensor转为list (如果之前没有做)
                    if not isinstance(prev_seq_tokens, list):
                        prev_seq_tokens = prev_seq_tokens.tolist()

                    for l in range(beam):  # 遍历新生成的beam个扩展
                        current_token_id = preds[k, j, 0, l].item()
                        current_prob = probs[k, j, 0, l].item()
                        current_logit_tensor = reshaped_logits[k, j, 0].unsqueeze(0)  # (1, vocab_size)

                        new_tokens_list = prev_seq_tokens + [current_token_id]
                        new_logits_list = prev_seq_logits + [current_logit_tensor] if prev_seq_logits else [
                            current_logit_tensor]

                        candidate = [
                            torch.tensor(new_tokens_list, device=self.device),  # 保持为tensor，方便下一步输入
                            new_logits_list,  # logit列表
                            prev_seq_prob * current_prob  # 更新概率
                        ]
                        all_candidates_for_sample_j.append(candidate)

                # 从所有新生成的候选序列中选出概率最高的beam个
                ordered_candidates = sorted(all_candidates_for_sample_j, key=lambda tup: tup[2], reverse=True)
                new_sequences_all_samples.append(ordered_candidates[:beam])
            sequences = new_sequences_all_samples  # 更新sequences

            # 检查是否所有最优序列都已生成结束标记
            all_finished = True
            temp_inputs_for_check = []
            for j in range(batch_size):
                # sequences[j][0][0] 是第j个样本当前最优序列的token ids (tensor)
                temp_inputs_for_check.append(sequences[j][0][0])
                if self.end_token_idx not in sequences[j][0][0]:
                    all_finished = False
            if all_finished:
                break

        # 提取最终结果 (每个样本最优的那个序列)
        final_logits_list = []
        final_inputs_list = []
        for j in range(batch_size):
            # sequences[j][0][1] 是最优序列的logits列表, sequences[j][0][0] 是最优序列的token ids
            # 将logits列表堆叠成一个tensor
            if sequences[j][0][1]:  # 确保logits列表不为空
                final_logits_list.append(torch.cat(sequences[j][0][1], dim=0))  # (seq_len, vocab_size)
            else:  # 如果序列为空（例如，如果beam=0或出错）
                final_logits_list.append(torch.empty(0, self.vocab_size, device=self.device))

            final_inputs_list.append(sequences[j][0][0])  # (seq_len)

        final_logits = torch.stack(final_logits_list) if final_logits_list else torch.empty(0, 0, self.vocab_size,
                                                                                            device=self.device)  # (bs, seq_len, vocab_size)
        final_inputs = torch.stack(final_inputs_list) if final_inputs_list else torch.empty(0, 0,
                                                                                            device=self.device)  # (bs, seq_len)

        return final_logits, final_inputs

    def converse(self, batch, mode):  # 对话模块 (包括训练和测试时的生成)
        """
        生成对话回复。
        context_tokens: (batch_size, context_len) 对话历史token序列
        context_entities: (batch_size, entity_len) 上下文实体序列
        context_words: (batch_size, word_len) 上下文词序列
        response: (batch_size, response_len) 目标回复序列 (仅在训练时提供)
        mode: 'train', 'valid', or 'test'
        """
        context_tokens, context_entities, context_words, response = batch  # 解包输入

        # 获取实体图和词图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 生成上下文实体和词的填充掩码
        entity_padding_mask = context_entities.eq(self.pad_entity_idx)
        word_padding_mask = context_words.eq(self.pad_word_idx)

        # 获取上下文实体和词的表示
        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        # 对上下文实体和词的表示应用自注意力机制，得到融合上下文的表示
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)  # (bs, kg_dim)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)  # (bs, kg_dim)

        # 使用Transformer编码器处理对话历史
        tokens_encoding = self.conv_encoder(context_tokens)  # ((encoder_outputs, hidden_state), mask)

        # 将KG的注意力表示进行线性变换，以匹配解码器期望的维度 (通常是ffn_size)
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)  # (bs, ffn_size)
        conv_word_emb = self.conv_word_attn_norm(word_attn_rep)  # (bs, ffn_size)
        # 将KG的原始表示也进行线性变换
        conv_entity_reps = self.conv_entity_norm(entity_representations)  # (bs, entity_len, ffn_size)
        conv_word_reps = self.conv_word_norm(word_representations)  # (bs, word_len, ffn_size)

        if mode != 'test':  # 训练或验证模式，使用Teacher Forcing
            logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)
            # logits: (bs, seq_len, vocab_size), preds: (bs, seq_len)
            # 计算损失
            logits_flatten = logits.view(-1, logits.shape[-1])  # (bs * seq_len, vocab_size)
            response_flatten = response.view(-1)  # (bs * seq_len)
            loss = self.conv_loss(logits_flatten, response_flatten)  # 计算交叉熵损失
            return loss, preds
        else:  # 测试模式，使用贪婪解码 (或Beam Search，这里用的是greedy)
            # 注意：KGSF论文中提到测试时也可能用beam search，这里代码实现的是greedy
            # 如果需要beam search，可以调用 _decode_beam_search_with_kg
            logits, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask)
            # preds: (bs, generated_seq_len)
            return preds  # 测试时只返回预测的token序列

    def forward(self, batch, stage, mode):  # 模型的前向传播主函数
        """
        根据不同的阶段 (stage) 调用相应的处理函数。
        batch: 输入数据批次
        stage: 'pretrain', 'rec' (推荐), 'conv' (对话)
        mode: 'train', 'valid', 'test'
        """
        if len(self.gpu) >= 2:  # 多GPU处理，确保图网络权重在当前GPU上
            # forward函数在不同的gpu上操作，图网络的权重需要复制到其他gpu上
            current_device = torch.cuda.current_device()
            self.entity_edge_idx = self.entity_edge_idx.to(current_device)
            self.entity_edge_type = self.entity_edge_type.to(current_device)
            self.word_edges = self.word_edges.to(current_device)
            # copy_mask也需要确保在当前设备
            if not self.copy_mask.device == current_device:
                self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                                 ).to(current_device)

        if stage == "pretrain":  # 预训练阶段
            loss = self.pretrain_infomax(batch)
        elif stage == "rec":  # 推荐阶段
            # recommend 函数返回 (rec_loss, info_loss, rec_scores)
            # 根据mode（训练/验证/测试），可能只需要部分返回值或进行不同处理
            rec_loss, info_loss, rec_scores = self.recommend(batch, mode)
            if mode == 'test':
                return rec_scores  # 测试时返回推荐得分
            else:
                # 训练和验证时返回总损失 (如果info_loss存在)
                loss = rec_loss
                if info_loss is not None:
                    loss += info_loss
                return loss  # 返回总损失，或者根据需要返回 (loss, rec_scores)
        elif stage == "conv":  # 对话阶段
            # converse 函数返回 (loss, preds) 或 preds
            output = self.converse(batch, mode)
            return output  # 返回损失和预测（训练/验证）或仅预测（测试）
        return None  # 如果stage不匹配，则返回None