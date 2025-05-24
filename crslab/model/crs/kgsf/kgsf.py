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
References:
    Zhou, Kun, et al. `"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`_ in KDD 2020.

.. _`"Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion."`:
   https://dl.acm.org/doi/abs/10.1145/3394486.3403143

"""
# 导入必要的库
import os # 操作系统接口模块

import numpy as np # 数值计算库
import torch # PyTorch深度学习框架
import torch.nn.functional as F # PyTorch神经网络函数库
from loguru import logger # 日志记录库
from torch import nn # PyTorch神经网络模块
from torch_geometric.nn import GCNConv, RGCNConv # PyTorch Geometric中的图卷积网络层

from crslab.config import MODEL_PATH # CRSLab中模型路径配置
from crslab.model.base import BaseModel # CRSLab模型基类
from crslab.model.utils.functions import edge_to_pyg_format # 图边格式转换工具函数
from crslab.model.utils.modules.attention import SelfAttentionSeq # 自注意力序列模块
from crslab.model.utils.modules.transformer import TransformerEncoder # Transformer编码器模块
from .modules import GateLayer, TransformerDecoderKG # KGSF模型自定义模块
from .resources import resources # KGSF模型特定资源


class KGSFModel(BaseModel):
    """
    KGSF模型类，继承自BaseModel。
    该模型通过融合知识图谱信息来提升对话推荐系统的性能。

    Attributes:
        vocab_size: 词汇表大小。
        pad_token_idx: padding标记的索引。
        start_token_idx: 开始标记的索引。
        end_token_idx: 结束标记的索引。
        token_emb_dim: 词元嵌入层的维度。
        pretrain_embedding: 预训练词嵌入的路径。
        n_word: 词汇知识图谱中词汇（概念）的数量。
        n_entity: 实体知识图谱中实体的数量。
        pad_word_idx: 词汇知识图谱中padding词汇的索引。
        pad_entity_idx: 实体知识图谱中padding实体的索引。
        num_bases: RGCN中用于参数共享的基的数量。
        kg_emb_dim: 知识图谱嵌入的维度。
        n_heads: Transformer中多头注意力的头数。
        n_layers: Transformer中的层数。
        ffn_size: Transformer中前馈神经网络的隐藏层大小。
        dropout: dropout比率。
        attention_dropout: Transformer注意力层的dropout比率。
        relu_dropout: Transformer中ReLU激活后的dropout比率。
        learn_positional_embeddings: 是否学习位置嵌入。
        embeddings_scale: 是否对嵌入进行缩放。
        reduction: Transformer编码器输出是否进行池化。
        n_positions: Transformer中位置嵌入的最大长度。
        response_truncate: 生成回复时的最大截断长度。
    """

    def __init__(self, opt, device, vocab, side_data):
        """
        KGSF模型初始化函数。

        Args:
            opt (dict): 包含超参数的字典。
            device (torch.device): 指定模型和数据存放的设备 (CPU或GPU)。
            vocab (dict): 包含词汇表信息的字典。
            side_data (dict): 包含辅助数据的字典，如知识图谱、预训练嵌入等。
        """
        self.opt = opt  # 存储opt以便在构建方法中访问
        self.device = device # 指定设备
        self.gpu = opt.get("gpu", [-1]) # 获取GPU配置，默认为CPU
        # 词汇表相关参数
        self.vocab_size = vocab['vocab_size'] # 词汇表大小
        self.pad_token_idx = vocab['pad'] # padding标记的索引
        self.start_token_idx = vocab['start'] # 开始标记的索引
        self.end_token_idx = vocab['end'] # 结束标记的索引
        self.token_emb_dim = opt['token_emb_dim'] # 词元嵌入维度
        self.pretrained_embedding = side_data.get('embedding', None) # 预训练词嵌入路径
        # 知识图谱相关参数
        self.n_word = vocab['n_word'] # 词汇知识图谱中词汇数量
        self.n_entity = vocab['n_entity'] # 实体知识图谱中实体数量
        self.pad_word_idx = vocab['pad_word'] # 词汇知识图谱padding词汇索引
        self.pad_entity_idx = vocab['pad_entity'] # 实体知识图谱padding实体索引
        entity_kg = side_data['entity_kg'] # 实体知识图谱数据
        self.n_relation = entity_kg['n_relation'] # 实体知识图谱中关系数量
        entity_edges = entity_kg['edge'] # 实体知识图谱的边
        # 将实体知识图谱的边转换为PyTorch Geometric格式
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device) # 边索引移至指定设备
        self.entity_edge_type = self.entity_edge_type.to(device) # 边类型移至指定设备
        word_edges = side_data['word_kg']['edge'] # 词汇知识图谱的边

        # 将词汇知识图谱的边转换为PyTorch Geometric格式
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)

        self.num_bases = opt['num_bases'] # RGCN的基数量
        self.kg_emb_dim = opt['kg_emb_dim'] # 知识图谱嵌入维度
        # Transformer相关参数
        self.n_heads = opt['n_heads'] # 多头注意力头数
        self.n_layers = opt['n_layers'] # Transformer层数
        self.ffn_size = opt['ffn_size'] # 前馈网络隐藏层大小
        self.dropout = opt['dropout'] # dropout比率
        self.attention_dropout = opt['attention_dropout'] # 注意力层dropout比率
        self.relu_dropout = opt['relu_dropout'] # ReLU后dropout比率
        self.learn_positional_embeddings = opt['learn_positional_embeddings'] # 是否学习位置嵌入
        self.embeddings_scale = opt['embeddings_scale'] # 是否缩放嵌入
        self.reduction = opt['reduction'] # Transformer编码器输出是否池化
        self.n_positions = opt['n_positions'] # 最大位置编码长度
        self.response_truncate = opt.get('response_truncate', 20) # 回复截断长度
        # 复制机制掩码相关
        dataset = opt['dataset'] # 数据集名称
        dpath = os.path.join(MODEL_PATH, "kgsf", dataset) # 模型特定资源路径
        resource = resources[dataset] # 获取数据集特定资源
        super(KGSFModel, self).__init__(opt, device, dpath, resource) # 调用父类初始化

    def build_model(self):
        """构建KGSF模型的各个组件。"""
        self._init_embeddings() # 初始化嵌入层
        self._build_kg_layer() # 构建知识图谱层
        self._build_infomax_layer() # 构建Infomax预训练层
        self._build_recommendation_layer() # 构建推荐层
        self._build_conversation_layer() # 构建对话生成层

    def _init_embeddings(self):
        """初始化词元嵌入和词汇知识图谱节点嵌入。"""
        if self.pretrained_embedding is not None: # 如果使用预训练词向量
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else: # 否则，随机初始化词元嵌入
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5) # 正态分布初始化
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0) # padding部分初始化为0

        # 初始化词汇知识图谱节点嵌入
        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5) # 正态分布初始化
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0) # padding部分初始化为0

        logger.debug('[Finish init embeddings]') # 记录日志：完成嵌入层初始化

    def _build_kg_layer(self):
        """构建知识图谱编码层，包括实体KG编码器、词汇KG编码器和门控融合层。"""
        # 实体知识图谱编码器 (RGCN)
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        # 实体序列自注意力层
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # 词汇知识图谱编码器 (GCN)
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        # 词汇序列自注意力层
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # 门控融合层，用于融合实体和词汇知识表示
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logger.debug('[Finish build kg layer]') # 记录日志：完成知识图谱层构建

    def _build_infomax_layer(self):
        """构建Infomax预训练任务相关的层。"""
        # Infomax任务的线性变换层和偏置项
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        # Infomax损失函数 (均方误差损失)
        self.infomax_loss = nn.MSELoss(reduction='sum')

        logger.debug('[Finish build infomax layer]') # 记录日志：完成Infomax层构建

    def _build_recommendation_layer(self):
        """构建推荐模块相关的层。"""
        # 推荐模块的偏置项
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        # 获取推荐任务的标签平滑因子
        rec_label_smoothing = self.opt.get('rec_label_smoothing', 0.0)
        logger.info(f"Recommendation Label Smoothing Factor: {rec_label_smoothing}")
        # 推荐损失函数 (交叉熵损失，支持标签平滑)
        self.rec_loss = nn.CrossEntropyLoss(label_smoothing=rec_label_smoothing)


        logger.debug('[Finish build rec layer]') # 记录日志：完成推荐层构建

    def _build_conversation_layer(self):
        """构建对话生成模块，包括Transformer编码器、知识融合层和Transformer解码器。"""
        # 注册一个持久缓冲区，存储开始标记的张量
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        # 对话历史编码器 (TransformerEncoder)
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

        # 用于将知识图谱表示投影到Transformer隐藏层维度的线性层
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

        # 复制机制相关的线性层
        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim) # 融合三种信息（实体、词汇、对话隐状态）
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size) # 输出复制概率
        # 加载预计算的复制掩码，指示哪些词汇可以被复制
        self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                         ).to(self.device)

        # 对话解码器 (定制的TransformerDecoderKG，能够融合知识图谱信息)
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
        # 获取对话生成任务的标签平滑因子
        conv_label_smoothing = self.opt.get('conv_label_smoothing', 0.0)
        logger.info(f"Conversation Label Smoothing Factor: {conv_label_smoothing}")
        # 对话生成损失函数 (交叉熵损失，忽略padding，支持标签平滑)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx, label_smoothing=conv_label_smoothing)


        logger.debug('[Finish build conv layer]') # 记录日志：完成对话层构建

    def pretrain_infomax(self, batch):
        """
        执行Infomax预训练任务。目标是让词汇知识图谱的表示能够预测相关的实体。

        Args:
            batch (tuple): 包含词汇序列和对应实体标签的批次数据。
                           words: (batch_size, word_length)
                           entity_labels: (batch_size, n_entity)，多标签one-hot或multi-hot形式

        Returns:
            torch.Tensor or None: Infomax损失；如果批次中没有有效的实体标签，则返回None。
        """
        words, entity_labels = batch # 解包批次数据

        loss_mask = torch.sum(entity_labels) # 计算有效实体标签的总数，用于归一化损失
        if loss_mask.item() == 0: # 如果没有有效标签，则不计算损失
            return None

        # 获取实体图和词汇图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 根据输入的词汇索引获取词汇表示
        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)，生成padding掩码

        # 对词汇表示应用自注意力机制
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        # 将自注意力加权后的词汇表示通过线性层进行变换
        word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
        # 计算词汇表示预测实体分布的logits
        info_predict = F.linear(word_info_rep, entity_graph_representations, self.infomax_bias.bias)  # (bs, #entity)
        # 计算均方误差损失
        loss = self.infomax_loss(info_predict, entity_labels) / loss_mask # 归一化损失
        return loss

    def recommend(self, batch, mode):
        """
        执行推荐任务。

        Args:
            batch (tuple): 包含上下文实体、上下文词汇、用于Infomax的实体标签和目标推荐物品的批次数据。
                           context_entities: (batch_size, entity_length)
                           context_words: (batch_size, word_length)
                           entities: (batch_size, n_entity)，用于Infomax的实体标签
                           movie: (batch_size)，目标推荐物品的ID

        Returns:
            tuple: (推荐损失, Infomax损失, 推荐得分)
        """
        context_entities, context_words, entities, movie = batch # 解包批次数据

        # 获取实体图和词汇图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 生成padding掩码
        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        # 获取上下文实体和词汇的表示
        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        # 对实体和词汇表示应用自注意力机制
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        # 使用门控机制融合实体和词汇的注意力表示，得到用户表示
        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        # 计算推荐得分
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        # 计算推荐损失
        rec_loss = self.rec_loss(rec_scores, movie)

        # 计算Infomax辅助损失
        info_loss_mask = torch.sum(entities) # 计算有效实体标签总数
        if info_loss_mask.item() == 0: # 如果没有有效标签
            info_loss = None
        else:
            word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (bs, #entity)
            info_loss = self.infomax_loss(info_predict, entities) / info_loss_mask # 归一化损失

        return rec_loss, info_loss, rec_scores

    def freeze_parameters(self):
        """冻结指定模型的参数，在训练对话生成模块时调用，以避免影响预训练好的知识图谱和推荐模块。"""
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.infomax_norm, self.rec_bias]
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False # 设置参数的requires_grad属性为False

    def _starts(self, batch_size):
        """返回batch_size个开始标记，用于解码器的起始输入。"""
        return self.START.detach().expand(batch_size, 1) # detach()确保不参与梯度计算

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):
        """
        使用Teacher Forcing方式进行解码，同时融合知识图谱信息。
        用于训练阶段。

        Args:
            token_encoding (tuple): 对话历史编码器的输出 (encoder_output, encoder_mask)。
            entity_reps (torch.Tensor): 上下文实体的知识图谱表示 (bs, entity_len, kg_emb_dim)。
            entity_emb_attn (torch.Tensor): 上下文实体经过自注意力后的表示 (bs, kg_emb_dim)。
            entity_mask (torch.Tensor): 上下文实体的padding掩码 (bs, entity_len)。
            word_reps (torch.Tensor): 上下文词汇的知识图谱表示 (bs, word_len, kg_emb_dim)。
            word_emb_attn (torch.Tensor): 上下文词汇经过自注意力后的表示 (bs, kg_emb_dim)。
            word_mask (torch.Tensor): 上下文词汇的padding掩码 (bs, word_len)。
            response (torch.Tensor): 真实的目标回复序列 (bs, seq_len)。

        Returns:
            tuple: (解码器输出的logits (bs, seq_len, vocab_size), 预测的词元序列 (bs, seq_len))
        """
        batch_size, seq_len = response.shape # 获取批次大小和序列长度
        start = self._starts(batch_size) # 获取开始标记
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long() # 构造解码器输入，使用真实回复作为Teacher Forcing

        # Transformer解码器前向传播，融合知识图谱信息
        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)
        # 将实体和词汇的注意力表示扩展到与对话隐状态相同的序列长度
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        # 融合三种隐状态（实体、词汇、对话）用于复制机制
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, dim)

        # 计算复制机制的logits，并应用预计算的复制掩码
        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
            0)  # (bs, seq_len, vocab_size)
        # 计算从词汇表生成的logits
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        # 合并生成logits和复制logits
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1) # 获取预测的词元序列
        return sum_logits, preds

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):
        """
        使用贪心策略进行解码，同时融合知识图谱信息。
        用于评估和推断阶段。

        Args:
            (同 _decode_forced_with_kg，除了没有 response 参数)

        Returns:
            tuple: (解码器输出的logits (bs, response_truncate, vocab_size), 生成的词元序列 (bs, response_truncate))
        """
        batch_size = token_encoding[0].shape[0] # 获取批次大小
        inputs = self._starts(batch_size).long() # 初始化解码器输入为开始标记
        incr_state = None # 用于存储解码器增量状态（Transformer通常不需要显式传递）
        logits = [] # 存储每一步的logits
        for _ in range(self.response_truncate): # 循环生成直到达到最大长度
            # Transformer解码器前向传播
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)，只取最后一个时间步的输出
            # 准备复制机制所需的知识表示
            db_latent = entity_emb_attn.unsqueeze(1)
            concept_latent = word_emb_attn.unsqueeze(1)
            # 融合隐状态
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

            # 计算复制和生成logits
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            sum_logits = copy_logits + gen_logits
            preds = sum_logits.argmax(dim=-1).long() # 贪心选择概率最大的词元
            logits.append(sum_logits) # 保存当前步的logits
            inputs = torch.cat((inputs, preds), dim=1) #将预测词元加入下一轮输入

            # 检查是否所有序列都已生成结束标记
            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished: # 如果都结束了，则停止生成
                break
        logits = torch.cat(logits, dim=1) # 拼接所有时间步的logits
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):
        """
        使用束搜索策略进行解码，同时融合知识图谱信息。
        用于评估和推断阶段，通常能生成更高质量的回复。

        Args:
            (同 _decode_greedy_with_kg，增加了 beam 参数)
            beam (int): 束搜索的宽度。

        Returns:
            tuple: (最终选择序列的logits (bs, response_truncate, vocab_size), 最终选择的词元序列 (bs, response_truncate))
        """
        batch_size = token_encoding[0].shape[0] # 获取批次大小
        # 初始化输入为开始标记，并调整形状以适应束搜索 (beam, batch_size, seq_len) -> (candidate_num, batch_size, seq_len)
        # 初始时，每个样本只有一个候选序列，即开始标记
        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None # 解码器增量状态

        # sequences 存储每个样本的 beam 个候选序列，每个候选序列包含 [生成的词元列表, logits列表, 概率]
        sequences = [[[list(), list(), 1.0]]] * batch_size
        for i in range(self.response_truncate): # 循环生成
            # 当 i=1 时 (即生成第二个词元时)，需要将编码器输出和知识表示复制 beam 次，因为每个样本将有 beam 个候选
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
                entity_reps = entity_reps.repeat(beam, 1, 1)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)
                entity_mask = entity_mask.repeat(beam, 1)
                word_reps = word_reps.repeat(beam, 1, 1)
                word_emb_attn = word_emb_attn.repeat(beam, 1)
                word_mask = word_mask.repeat(beam, 1)

            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                inputs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        inputs.append(text)
                inputs = torch.stack(inputs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            with torch.no_grad():
                dialog_latent, incr_state = self.conv_decoder(
                    inputs.reshape(len(sequences[0]) * batch_size, -1), # len(sequences[0]) 是当前的候选数量
                    token_encoding, word_reps, word_mask,
                    entity_reps, entity_mask, incr_state
                )
                dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
                db_latent = entity_emb_attn.unsqueeze(1)
                concept_latent = word_emb_attn.unsqueeze(1)
                # 融合隐状态
                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                # 计算复制和生成logits
                copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
                sum_logits = copy_logits + gen_logits

            logits = sum_logits.reshape(len(sequences[0]), batch_size, 1, -1)
            # turn into probabilities,in case of negative numbers
            probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)

            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        logit = sequences[j][n][1]
                        if logit == []:
                            logit_tmp = logits[n][j][0].unsqueeze(0)
                        else:
                            logit_tmp = torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((inputs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            # We need to check the sequences generated so far
            all_finished_this_beam = True
            current_inputs_for_check = torch.stack([s[0][0] for s in sequences]) # Get current sequences from beam
            if current_inputs_for_check.numel() > 0 : # Ensure tensor is not empty
                 all_finished = ((current_inputs_for_check == self.end_token_idx).sum(dim=-1) > 0).all().item() # Check if all sequences in beam have END
                 if all_finished:
                    break
        logits = torch.stack([seq[0][1] for seq in sequences])
        inputs = torch.stack([seq[0][0] for seq in sequences])
        return logits, inputs


    def converse(self, batch, mode):
        """
        执行对话生成任务。

        Args:
            batch (tuple): 包含对话上下文、实体、词汇和目标回复的批次数据。
            mode (str): 'train', 'valid' 或 'test'。

        Returns:
            tuple or torch.Tensor: 训练/验证模式下返回 (损失, 预测序列)，测试模式下返回预测序列。
        """
        context_tokens, context_entities, context_words, response = batch # 解包批次数据

        # 获取实体图和词汇图的节点表示
        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        # 生成padding掩码
        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, seq_len)

        # 获取上下文实体和词汇的表示
        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        # 对实体和词汇表示应用自注意力机制
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

        # 对话历史编码
        tokens_encoding = self.conv_encoder(context_tokens)
        # 将知识表示投影到与Transformer兼容的维度
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)
        conv_word_emb = self.conv_word_attn_norm(word_attn_rep)
        conv_entity_reps = self.conv_entity_norm(entity_representations)
        conv_word_reps = self.conv_word_norm(word_representations)

        if mode != 'test': # 训练或验证模式
            # 使用Teacher Forcing进行解码
            logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)

            # 计算损失
            logits = logits.view(-1, logits.shape[-1]) # (bs * seq_len, vocab_size)
            response = response.view(-1) # (bs * seq_len)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else: # 测试模式
            # 使用贪心策略进行解码
            logits, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask)
            return preds # 测试时通常只关心生成的序列

    def forward(self, batch, stage, mode):
        """
        模型的前向传播函数，根据阶段（预训练、推荐、对话）调用相应的处理函数。

        Args:
            batch (tuple): 输入的批次数据。
            stage (str): 当前的执行阶段 ('pretrain', 'rec', 'conv')。
            mode (str): 当前的执行模式 ('train', 'valid', 'test')。

        Returns:
            torch.Tensor or tuple: 根据阶段和模式返回损失或预测结果。
        """
        # 如果使用多GPU，需要确保图相关的张量在当前GPU上
        if len(self.gpu) >= 2:
            self.entity_edge_idx = self.entity_edge_idx.cuda(torch.cuda.current_device())
            self.entity_edge_type = self.entity_edge_type.cuda(torch.cuda.current_device())
            self.word_edges = self.word_edges.cuda(torch.cuda.current_device())
            self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                             ).cuda(torch.cuda.current_device())
        # 根据阶段调用不同的处理函数
        if stage == "pretrain": # 预训练阶段
            loss = self.pretrain_infomax(batch)
        elif stage == "rec": # 推荐阶段
            loss = self.recommend(batch, mode)
        elif stage == "conv": # 对话生成阶段
            loss = self.converse(batch, mode)
        return loss
