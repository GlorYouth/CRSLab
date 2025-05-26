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

import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger  # 日志库
from torch import nn
from torch_geometric.nn import GCNConv, FastRGCNConv  # 图神经网络层
# 导入 AMP 相关模块
from torch.cuda.amp import autocast, GradScaler  # 用于混合精度训练

from crslab.config import MODEL_PATH  # 模型路径配置
from crslab.model.base import BaseModel  # 模型基类
from crslab.model.utils.functions import edge_to_pyg_format  # 边转换工具
from crslab.model.utils.modules.attention import SelfAttentionSeq  # 自注意力模块
from crslab.model.utils.modules.transformer import TransformerEncoder  # Transformer 编码器
from .modules import GateLayer, TransformerDecoderKG  # 自定义模块
from .resources import resources  # 资源文件


class KGSFModel(BaseModel):
    """

    属性:
        vocab_size: 词汇表大小
        pad_token_idx: padding token 的 id
        start_token_idx: start token 的 id
        end_token_idx: end token 的 id
        token_emb_dim: token 嵌入层的维度
        pretrain_embedding: 预训练词向量路径
        n_word: 词语数量
        n_entity: 实体数量
        pad_word_idx: 词语 padding 的 id
        pad_entity_idx: 实体 padding 的 id
        num_bases: RGCN 的基数
        kg_emb_dim: 知识图谱嵌入维度
        n_heads: Transformer 的头数
        n_layers: Transformer 的层数
        ffn_size: Transformer FFN 层的隐藏层大小
        dropout: dropout 率
        attention_dropout: attention 层的 dropout 率
        relu_dropout: relu 层的 dropout 率
        learn_positional_embeddings: 是否学习位置嵌入
        embeddings_scale: 是否缩放嵌入
        reduction: 是否使用 reduction
        n_positions: 位置数量
        response_truncate: 生成回复的最大长度
        use_amp: 是否使用混合精度训练
        scaler: GradScaler 对象，用于混合精度训练

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        参数:
            opt (dict): 超参数字典
            device (torch.device): 指定数据和模型存放的设备
            vocab (dict): 词汇表信息字典
            side_data (dict): 边数据字典

        """
        self.device = device
        self.gpu = opt.get("gpu", [-1])  # 获取GPU配置，默认为-1 (表示CPU)
        # 词汇表相关参数
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # 知识图谱相关参数
        self.n_word = vocab['n_word']
        self.n_entity = vocab['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']  # 实体知识图谱
        self.n_relation = entity_kg['n_relation']  # 关系数量
        entity_edges = entity_kg['edge']  # 实体图的边
        # 将边转换为 PyG 格式
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)  # 边索引移至设备
        self.entity_edge_type = self.entity_edge_type.to(device)  # 边类型移至设备
        word_edges = side_data['word_kg']['edge']  # 词知识图谱的边

        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)  # 词图边转换为PyG格式并移至设备

        self.num_bases = opt['num_bases']  # RGCN 的基数
        self.kg_emb_dim = opt['kg_emb_dim']  # KG 嵌入维度
        # Transformer 相关参数
        self.n_heads = opt['n_heads']  # 头数
        self.n_layers = opt['n_layers']  # 层数
        self.ffn_size = opt['ffn_size']  # FFN 层大小
        self.dropout = opt['dropout']  # Dropout 率
        self.attention_dropout = opt['attention_dropout']  # Attention Dropout 率
        self.relu_dropout = opt['relu_dropout']  # ReLU Dropout 率
        self.learn_positional_embeddings = opt['learn_positional_embeddings']  # 是否学习位置嵌入
        self.embeddings_scale = opt['embeddings_scale']  # 是否缩放嵌入
        self.reduction = opt['reduction']  # 是否使用规约
        self.n_positions = opt['n_positions']  # 最大位置数
        self.response_truncate = opt.get('response_truncate', 20)  # 回复截断长度
        # 复制掩码相关
        dataset = opt['dataset']  # 数据集名称
        dpath = os.path.join(MODEL_PATH, "kgsf", dataset)  # 数据路径
        resource = resources[dataset]  # 资源
        self.dpath = dpath  # 保存 dpath 以便后续使用

        # 初始化 AMP (混合精度训练) 相关组件
        self.use_amp = self.device.type == 'cuda'  # 判断是否使用 CUDA，从而决定是否启用 AMP
        self.scaler = GradScaler(enabled=self.use_amp)  # 初始化 GradScaler，仅在 CUDA 环境下启用
        logger.info(f"[AMP] 混合精度训练已{'启用' if self.use_amp else '禁用'}.")

        super(KGSFModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        # 构建模型的各个组件
        self._init_embeddings()  # 初始化词向量嵌入
        self._build_kg_layer()  # 构建知识图谱层
        self._build_infomax_layer()  # 构建 Infomax 层
        self._build_recommendation_layer()  # 构建推荐层
        self._build_conversation_layer()  # 构建对话层

    def _init_embeddings(self):
        # 初始化词向量和知识图谱中词的嵌入
        if self.pretrained_embedding is not None:
            # 如果有预训练词向量，则加载
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
            logger.info("[词嵌入] 已加载预训练词嵌入.")
        else:
            # 否则，随机初始化
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.token_emb_dim ** -0.5)  # 使用正态分布初始化
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)  # padding token 初始化为0
            logger.info("[词嵌入] 从头开始初始化词嵌入.")

        # 初始化词知识图谱嵌入
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)  # 使用正态分布初始化
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)  # padding word 初始化为0
        logger.info("[词嵌入] 已初始化词知识图谱嵌入.")
        logger.debug('[完成嵌入初始化]')

    def _build_kg_layer(self):
        # 构建知识图谱相关的编码器和注意力机制
        # 实体编码器
        self.entity_encoder = FastRGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 实体自注意力

        # 概念编码器
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 词语自注意力

        # 门控机制
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logger.debug('[完成知识图谱层构建]')

    def _build_infomax_layer(self):
        # 构建 Infomax 相关的层，用于预训练
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)  # 归一化层
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 偏置层 (注意：原代码的 bias 是一个 Linear 层)
        self.infomax_loss = nn.MSELoss(reduction='sum')  # Infomax 损失函数 (均方误差损失)

        logger.debug('[完成 Infomax 层构建]')

    def _build_recommendation_layer(self):
        # 构建推荐相关的层
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 推荐偏置层 (注意：原代码的 bias 是一个 Linear 层)
        self.rec_loss = nn.CrossEntropyLoss()  # 推荐任务的损失函数 (交叉熵损失)

        logger.debug('[完成推荐层构建]')

    def _build_conversation_layer(self):
        # 构建对话生成相关的层
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))  # 注册起始符
        # 对话编码器
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.reduction,
            n_positions=self.n_positions,
        )

        # 用于融合知识信息的线性层
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 实体嵌入归一化
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 实体注意力归一化
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 词嵌入归一化
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)  # 词注意力归一化

        # 复制机制相关的层
        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)  # 复制机制归一化
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)  # 复制机制输出层
        # 加载复制掩码，确保只复制词汇表中的特定token
        copy_mask_path = os.path.join(self.dpath, "copy_mask.npy")
        if os.path.exists(copy_mask_path):
            self.copy_mask = torch.as_tensor(np.load(copy_mask_path).astype(bool)).to(self.device)
            logger.info(f"[复制机制] 已从 {copy_mask_path} 加载复制掩码")
        else:
            logger.warning(f"[复制机制] 在 {copy_mask_path} 未找到 copy_mask.npy。将禁用复制掩码。")
            # 创建一个全为 False 的掩码，或者根据需要调整
            self.copy_mask = torch.zeros(self.vocab_size, dtype=torch.bool).to(self.device)

        # 对话解码器 (集成知识图谱的 Transformer Decoder)
        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)  # 对话生成的损失函数 (交叉熵损失，忽略padding)

        logger.debug('[完成对话层构建]')

    def pretrain_infomax(self, batch):
        """
        Infomax 预训练阶段
        参数:
            batch: 包含 'words' 和 'entity_labels' 的元组
                words (torch.Tensor): (batch_size, word_length) 输入的词语序列
                entity_labels (torch.Tensor): (batch_size, n_entity) 实体标签，用于计算 Infomax 损失
        返回:
            torch.Tensor or None: 计算得到的损失值，如果无有效标签则为 None
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            words, entity_labels = batch

            loss_mask = torch.sum(entity_labels)  # 计算有效的标签数量，用于归一化损失
            if loss_mask.item() == 0:  # 如果没有有效标签，则不计算损失
                logger.warning("[Infomax预训练] 当前批次无实体标签，跳过损失计算。")
                return None

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 获取当前批次词语的表示，并处理 padding
            word_representations = word_graph_representations[words]
            word_padding_mask = words.eq(self.pad_word_idx)  # (batch_size, seq_len) 词语padding掩码

            # 通过自注意力机制聚合词语表示
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
            word_info_rep = self.infomax_norm(word_attn_rep)  # (batch_size, dim) 规范化后的词语信息表示

            # 预测实体，计算 Infomax 损失
            # F.linear(input, weight, bias) 等价于 input.matmul(weight.t()) + bias
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (batch_size, n_entity) 预测的实体分布
            loss = self.infomax_loss(info_predict, entity_labels) / loss_mask  # 计算并归一化损失

        # loss 返回后，在训练循环中需要使用 scaler 进行缩放: scaler.scale(loss).backward()
        return loss

    def recommend(self, batch, mode):
        """
        推荐阶段
        参数:
            batch: 包含 'context_entities', 'context_words', 'entities', 'movie' 的元组
                context_entities (torch.Tensor): (batch_size, entity_length) 上下文中的实体
                context_words (torch.Tensor): (batch_size, word_length) 上下文中的词语
                entities (torch.Tensor): (batch_size, n_entity) 推荐相关的实体标签 (用于 Infomax 辅助损失)
                movie (torch.Tensor): (batch_size) 目标推荐的电影/项目 ID
            mode (str): 当前模式 ('train', 'val', 'test')
        返回:
            tuple: (rec_loss, info_loss, rec_scores)
                rec_loss (torch.Tensor): 推荐损失
                info_loss (torch.Tensor or None): Infomax 辅助损失，如果无有效标签则为 None
                rec_scores (torch.Tensor): 推荐得分 (batch_size, n_entity)
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            context_entities, context_words, entities, movie = batch

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 处理 padding
            entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (batch_size, entity_len) 实体padding掩码
            word_padding_mask = context_words.eq(self.pad_word_idx)  # (batch_size, word_len) 词语padding掩码

            # 获取上下文实体和词语的表示
            entity_representations = entity_graph_representations[context_entities]
            word_representations = word_graph_representations[context_words]

            # 通过自注意力机制聚合表示
            entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)  # 上下文实体表示
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)  # 上下文词语表示

            # 通过门控机制融合实体和词语表示，得到用户表示
            user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)  # 用户表示
            # 计算推荐得分
            rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (batch_size, n_entity)

            # 计算推荐损失
            rec_loss = self.rec_loss(rec_scores, movie)

            # 计算辅助的 Infomax 损失
            info_loss_mask = torch.sum(entities)  # 有效实体标签数量
            if info_loss_mask.item() == 0:
                info_loss = None  # 如果没有有效实体标签，则不计算此辅助损失
                logger.debug("[推荐] 当前批次无用于辅助Infomax损失的实体标签。")
            else:
                word_info_rep = self.infomax_norm(word_attn_rep)  # (batch_size, dim)
                info_predict = F.linear(word_info_rep, entity_graph_representations,
                                        self.infomax_bias.bias)  # (batch_size, n_entity)
                info_loss = self.infomax_loss(info_predict, entities) / info_loss_mask

        # rec_loss 和 info_loss 返回后，在训练循环中需要使用 scaler 进行缩放
        # 例如: total_loss = rec_loss + alpha * info_loss (如果 info_loss 不为 None)
        # scaler.scale(total_loss).backward()
        return rec_loss, info_loss, rec_scores

    def freeze_parameters(self):
        # 冻结指定模型的参数，使其在训练中不更新
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.infomax_norm, self.rec_bias]
        logger.info("[冻结参数] 正在冻结指定层的参数。")
        for model_idx, model in enumerate(freeze_models):
            if model is None:
                logger.warning(f"索引 {model_idx} 处的模型为 None，跳过冻结。")
                continue
            for param_idx, p in enumerate(model.parameters()):
                p.requires_grad = False  # 设置参数不需要梯度
            logger.debug(f"已冻结模型参数: {model.__class__.__name__}")

    def _starts(self, batch_size):
        """返回 batch_size 个起始符"""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):
        # 使用 Teacher Forcing 进行解码 (主要用于训练)
        # 参数:
        #   token_encoding: 上下文编码器的输出 (通常是元组，包含encoder_outputs, encoder_hidden)
        #   entity_reps: 上下文实体表示 (batch_size, num_context_entities, kg_emb_dim)
        #   entity_emb_attn: 上下文实体注意力加权表示 (batch_size, kg_emb_dim)
        #   entity_mask: 上下文实体掩码 (batch_size, num_context_entities)
        #   word_reps: 上下文词语表示 (batch_size, num_context_words, kg_emb_dim)
        #   word_emb_attn: 上下文词语注意力加权表示 (batch_size, kg_emb_dim)
        #   word_mask: 上下文词语掩码 (batch_size, num_context_words)
        #   response: 目标回复序列 (batch_size, response_len)
        # 返回:
        #   sum_logits: 解码器输出的 logits (batch_size, response_len, vocab_size)
        #   preds: 根据 logits 预测的 token (batch_size, response_len)

        batch_size, seq_len = response.shape
        # 构建解码器输入，将起始符与真实回复拼接 (去掉最后一个token)
        start = self._starts(batch_size)  # (batch_size, 1)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()  # (batch_size, response_len)

        # 通过对话解码器获取解码结果
        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (batch_size, seq_len, dim)

        # 准备用于复制机制的实体和词语隐状态
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, dim)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, dim)

        # 拼接并进行线性变换，得到复制机制的隐状态
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (batch_size, seq_len, dim)

        # 计算复制概率和生成概率
        # 复制概率仅限于 copy_mask 中指定的词
        copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1,
                                                                             -1)  # (batch_size, seq_len, vocab_size)
        # 生成概率通过解码器的隐状态直接映射到词汇表
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (batch_size, seq_len, vocab_size)

        # 合并复制和生成概率
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1)  # 预测的 token
        return sum_logits, preds

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):
        # 使用 Greedy Search 进行解码 (主要用于推理/测试)
        # 参数: (同 _decode_forced_with_kg, 但没有 response 参数)
        # 返回:
        #   logits: 生成序列每一步的 logits (batch_size, generated_seq_len, vocab_size)
        #   inputs: 生成的 token 序列 (batch_size, generated_seq_len)

        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()  # 解码器初始输入为起始符 (batch_size, 1)
        incr_state = None  # 用于存储解码器每一步的状态，实现增量解码
        logits_list = []  # 存储每一步的 logits

        for _ in range(self.response_truncate):  # 最多生成 response_truncate 长度的回复
            # 单步解码
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (batch_size, 1, dim) 只取最后一个时间步的输出

            # 准备复制机制的隐状态
            db_latent = entity_emb_attn.unsqueeze(1)  # (batch_size, 1, dim)
            concept_latent = word_emb_attn.unsqueeze(1)  # (batch_size, 1, dim)
            copy_latent = self.copy_norm(
                torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))  # (batch_size, 1, dim)

            # 计算复制和生成概率
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1,
                                                                                 -1)  # (batch_size, 1, vocab_size)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (batch_size, 1, vocab_size)
            sum_logits = copy_logits + gen_logits  # (batch_size, 1, vocab_size)

            preds = sum_logits.argmax(dim=-1).long()  # (batch_size, 1) 贪心选择概率最大的 token
            logits_list.append(sum_logits)  # 存储 logits
            inputs = torch.cat((inputs, preds), dim=1)  # 将预测的 token 加入到下一次的输入

            # 检查是否所有批次的样本都已生成结束符
            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break

        logits = torch.cat(logits_list, dim=1)  # (batch_size, generated_seq_len, vocab_size)
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):
        # 使用 Beam Search 进行解码 (用于提升生成质量，但计算量较大)
        # 参数: (同 _decode_greedy_with_kg, 额外有 beam 参数)
        #   beam (int): beam search 的宽度
        # 返回:
        #   None: 当前实现不返回完整的 logits 序列以匹配原接口，可根据需要修改
        #   final_preds_list: 一个列表，包含 batch 中每个样本的最佳预测 token 序列 (torch.Tensor)
        # 注意：此处的 Beam Search 实现较为复杂，且在混合精度下可能需要更仔细的调优

        batch_size = token_encoding[0].shape[0]

        # 初始化 beam
        # sequences 列表的每个元素对应 batch 中的一个样本
        # 每个样本的元素是一个列表，包含 beam 个候选：元组 (tokens_list, logits_list, log_probability)
        sequences = [[(self._starts(1).squeeze().tolist(), [], 0.0)] for _ in range(batch_size)]  # 使用 0.0 作为初始 log 概率

        for _ in range(self.response_truncate):  # 迭代生成 token，直到达到最大长度
            # 存储当前解码步骤中所有活跃 beam 的输入及相关信息
            current_inputs_list = []
            current_token_enc_list_0 = []  # 存储 token_encoding 元组的第一部分
            current_token_enc_list_1 = []  # 存储 token_encoding 元组的第二部分
            current_ent_reps_list = []
            current_ent_attn_list = []
            current_ent_mask_list = []
            current_word_reps_list = []
            current_word_attn_list = []
            current_word_mask_list = []

            # active_beam_references 存储 (原始批次索引, 该批次中 beam 数据的索引)
            active_beam_references = []

            for i in range(batch_size):  # 遍历 batch 中的每个样本
                for k_idx, (tokens, _, prob) in enumerate(sequences[i]):  # 遍历当前样本的 beam 个候选
                    # 如果序列已结束 (最后一个 token 是结束符且长度大于1)，则不再扩展
                    if tokens and tokens[-1] == self.end_token_idx and len(tokens) > 1:
                        continue  # 跳过对此已完成 beam 的扩展

                    active_beam_references.append((i, k_idx))  # 记录此活跃 beam 的引用

                    # 准备当前活跃 beam 的输入数据
                    current_inputs_list.append(torch.tensor(tokens, device=self.device).unsqueeze(0))
                    current_token_enc_list_0.append(token_encoding[0][i:i + 1])
                    current_token_enc_list_1.append(token_encoding[1][i:i + 1])
                    current_ent_reps_list.append(entity_reps[i:i + 1])
                    current_ent_attn_list.append(entity_emb_attn[i:i + 1])
                    current_ent_mask_list.append(entity_mask[i:i + 1])
                    current_word_reps_list.append(word_reps[i:i + 1])
                    current_word_attn_list.append(word_emb_attn[i:i + 1])
                    current_word_mask_list.append(word_mask[i:i + 1])

            if not current_inputs_list:  # 如果所有 beam 都已结束 (没有活跃的 beam 了)
                break

            # 批量处理所有活跃的 beam
            inputs_batched = torch.cat(current_inputs_list, dim=0)
            token_encoding_batched = (torch.cat(current_token_enc_list_0, dim=0),
                                      torch.cat(current_token_enc_list_1, dim=0))
            entity_reps_batched = torch.cat(current_ent_reps_list, dim=0)
            entity_emb_attn_batched = torch.cat(current_ent_attn_list, dim=0)
            entity_mask_batched = torch.cat(current_ent_mask_list, dim=0)
            word_reps_batched = torch.cat(current_word_reps_list, dim=0)
            word_emb_attn_batched = torch.cat(current_word_attn_list, dim=0)
            word_mask_batched = torch.cat(current_word_mask_list, dim=0)

            # 解码器前向传播 (为简化，incr_state 设为 None，表示非增量解码)
            dialog_latent, _ = self.conv_decoder(inputs_batched, token_encoding_batched,
                                                 word_reps_batched, word_mask_batched,
                                                 entity_reps_batched, entity_mask_batched,
                                                 None)
            dialog_latent = dialog_latent[:, -1:, :]  # 获取最后一个时间步的输出

            # 计算复制和生成 logits
            db_latent = entity_emb_attn_batched.unsqueeze(1)
            concept_latent = word_emb_attn_batched.unsqueeze(1)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1, -1)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            sum_logits = copy_logits + gen_logits  # (num_active_beams, 1, vocab_size)

            # 计算 log 概率并取 top-k
            log_probs = F.log_softmax(sum_logits.squeeze(1), dim=-1)  # (num_active_beams, vocab_size)
            top_log_probs, top_indices = log_probs.topk(beam, dim=-1)  # (num_active_beams, beam)

            # 临时存储下一步的 beam 候选，修剪前
            temp_sequences = [[] for _ in range(batch_size)]

            active_beam_counter = 0  # 追踪已处理的活跃 beam
            for i in range(batch_size):  # 遍历原始批次中的每个样本
                # 将当前样本已完成的序列添加到 temp_sequences
                for k_idx, (tokens, stored_logits, prob) in enumerate(sequences[i]):
                    if tokens and tokens[-1] == self.end_token_idx and len(tokens) > 1:
                        temp_sequences[i].append((list(tokens), stored_logits, prob))

                # 将当前样本扩展的序列添加到 temp_sequences
                for (orig_batch_idx, beam_data_idx) in active_beam_references:
                    if orig_batch_idx == i:  # 如果此活跃 beam 属于当前处理的批次样本
                        tokens, _, current_prob = sequences[orig_batch_idx][beam_data_idx]
                        for j in range(beam):  # 对于此活跃 beam 的 beam 个扩展
                            new_token = top_indices[active_beam_counter, j].item()
                            new_log_prob_token = top_log_probs[active_beam_counter, j].item()

                            new_tokens_list = list(tokens) + [new_token]
                            new_total_prob = current_prob + new_log_prob_token  # log 概率相加
                            # 存储 logits 较为复杂，这里简化为存储空列表
                            temp_sequences[i].append((new_tokens_list, [], new_total_prob))
                        active_beam_counter += 1

            # 对每个批次样本的 beam 候选进行修剪
            for i in range(batch_size):
                if temp_sequences[i]:
                    ordered = sorted(temp_sequences[i], key=lambda tup: tup[2], reverse=True)  # 按 log 概率降序排序
                    sequences[i] = ordered[:beam]  # 保留 top beam 个候选
                elif not sequences[i]:  # 如果 temp_sequences[i] 为空且 sequences[i] 也为空 (例如第一步解码失败)
                    logger.warning(f"Beam search 第 {i} 项无有效候选。重新初始化为起始符。")
                    sequences[i] = [(self._starts(1).squeeze().tolist(), [], -float('inf'))]  # 赋予一个极低的 log 概率

            # 检查是否所有批次样本的 top beam 都已结束
            all_finished_flag = True
            for i in range(batch_size):
                if not sequences[i] or not sequences[i][0][0] or sequences[i][0][0][-1] != self.end_token_idx:
                    all_finished_flag = False
                    break
            if all_finished_flag:
                break

        # 为每个批次样本选择最佳序列
        final_preds_list = []
        for i in range(batch_size):
            if sequences[i] and sequences[i][0][0]:  # 检查是否存在有效的最佳序列
                best_sequence_tokens = sequences[i][0][0]
                # 如果序列达到最大长度但未自然结束，则添加结束符
                if best_sequence_tokens[-1] != self.end_token_idx and len(
                        best_sequence_tokens) == self.response_truncate:
                    best_sequence_tokens.append(self.end_token_idx)
                final_preds_list.append(torch.tensor(best_sequence_tokens, device=self.device))
            else:  # 如果未找到有效序列 (罕见情况)，则回退
                logger.warning(f"Beam search 第 {i} 项未产生有效的最佳序列。使用结束符作为回退。")
                final_preds_list.append(torch.tensor([self.end_token_idx], device=self.device))

        # 原函数期望返回 (logits, inputs)
        # beam search 返回实际的 logits 比较复杂，这里返回 None 和预测的 token 序列列表
        return None, final_preds_list

    def converse(self, batch, mode):
        """
        对话阶段
        参数:
            batch: 包含 'context_tokens', 'context_entities', 'context_words', 'response' (训练时) 的元组
                context_tokens (torch.Tensor): (batch_size, context_length) 上下文 token
                context_entities (torch.Tensor): (batch_size, entity_length) 上下文实体
                context_words (torch.Tensor): (batch_size, word_length) 上下文词语
                response_or_none (torch.Tensor or None): 目标回复 (训练时为 Tensor, 测试时为 None)
            mode (str): 当前模式 ('train', 'val', 'test')
        返回:
            tuple or torch.Tensor:
                训练/验证模式: (loss, preds) -> 损失和预测的 token 序列
                测试模式: preds -> 预测的 token 序列
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            context_tokens, context_entities, context_words, response_or_none = batch  # response 在测试模式下可能为 None

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 处理 padding
            entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (batch_size, entity_len)
            word_padding_mask = context_words.eq(self.pad_word_idx)  # (batch_size, seq_len)

            # 获取上下文实体和词语的表示
            entity_representations = entity_graph_representations[context_entities]
            word_representations = word_graph_representations[context_words]

            # 通过自注意力机制聚合表示
            entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

            # 对话编码器对上下文 token 进行编码
            tokens_encoding = self.conv_encoder(context_tokens)

            # 准备用于解码器的知识图谱相关嵌入
            conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)  # 融合上下文的实体表示 (注意力加权后)
            conv_word_emb = self.conv_word_attn_norm(word_attn_rep)  # 融合上下文的词语表示 (注意力加权后)
            conv_entity_reps = self.conv_entity_norm(entity_representations)  # 上下文中的原始实体表示
            conv_word_reps = self.conv_word_norm(word_representations)  # 上下文中的原始词语表示

            if mode != 'test':
                # 训练或验证模式，response_or_none 应该是实际的 response tensor
                response = response_or_none
                logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                            entity_padding_mask,
                                                            conv_word_reps, conv_word_emb, word_padding_mask,
                                                            response)

                # 重塑 logits 和 response 以计算损失
                logits_flat = logits.reshape(-1, logits.shape[-1])  # (batch_size * seq_len, vocab_size)
                response_flat = response.reshape(-1)  # (batch_size * seq_len)
                loss = self.conv_loss(logits_flat, response_flat)  # 计算损失
                # loss 返回后，在训练循环中需要使用 scaler 进行缩放: scaler.scale(loss).backward()
                return loss, preds
            else:  # 测试模式
                # 测试时首选 Beam Search (如果已实现且稳定)
                # _, preds = self._decode_beam_search_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                #                                           entity_padding_mask,
                #                                           conv_word_reps, conv_word_emb, word_padding_mask,
                #                                           beam=4) # beam 大小示例
                # 目前根据原始 'test' 模式的结构，继续使用 greedy search
                _, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                       entity_padding_mask,
                                                       conv_word_reps, conv_word_emb, word_padding_mask)
                return preds  # 测试模式通常只返回预测结果

    def forward(self, batch, stage, mode):
        """
        模型的前向传播入口
        参数:
            batch: 当前批次的数据
            stage (str): 当前阶段 ('pretrain', 'rec', 'conv')
            mode (str): 当前模式 ('train', 'val', 'test')
        返回:
            根据 stage 和 mode 返回不同的输出 (损失或预测结果)
        """
        # 多 GPU 处理：确保图网络的边索引和类型在当前 CUDA 设备上
        if len(self.gpu) >= 2 and self.device.type == 'cuda':
            current_cuda_device = torch.cuda.current_device()
            self.entity_edge_idx = self.entity_edge_idx.to(current_cuda_device)
            self.entity_edge_type = self.entity_edge_type.to(current_cuda_device)
            self.word_edges = self.word_edges.to(current_cuda_device)

            copy_mask_path = os.path.join(self.dpath, "copy_mask.npy")
            if hasattr(self, 'copy_mask') and self.copy_mask is not None:  # 确保 copy_mask 已加载
                self.copy_mask = self.copy_mask.to(current_cuda_device)
            elif os.path.exists(copy_mask_path):  # 尝试重新加载并移动到当前设备
                self.copy_mask = torch.as_tensor(np.load(copy_mask_path).astype(bool)).to(current_cuda_device)
            # 否则，copy_mask 可能未正确初始化或不存在，在 _build_conversation_layer 中会有警告

        # 根据不同阶段调用相应的处理函数
        if stage == "pretrain":
            loss = self.pretrain_infomax(batch)
            # 预训练阶段，通常只返回一个损失值
            # 调用者 (例如 system.step) 会处理这个损失
            return loss
        elif stage == "rec":
            # self.recommend 返回: rec_loss, info_loss, rec_scores
            rec_loss, info_loss, rec_scores = self.recommend(batch, mode)
            # 在推荐阶段，KGSFSystem 的 step 方法期望解包 (rec_loss, info_loss, rec_predict)
            # rec_predict 对应于此处的 rec_scores
            return rec_loss, info_loss, rec_scores

        elif stage == "conv":
            if mode != 'test':
                # self.converse 在非测试模式下返回 (loss, preds)
                loss, preds = self.converse(batch, mode)
                # KGSFSystem 的 step 方法在非测试对话阶段期望解包 (loss, preds)
                return loss, preds
            else:
                # self.converse 在测试模式下返回 preds
                preds = self.converse(batch, mode)
                # KGSFSystem 的 step 方法在测试对话阶段期望 preds
                return preds
        else:
            logger.error(f"未知的阶段: {stage}")
            raise ValueError(f"未知的阶段: {stage}")
