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

import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, FastRGCNConv
# 导入 AMP 相关模块
from torch.cuda.amp import autocast, GradScaler

from crslab.config import MODEL_PATH
from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from .modules import GateLayer, TransformerDecoderKG
from .resources import resources


class KGSFModel(BaseModel):
    """

    Attributes:
        vocab_size: 词汇表大小 (A integer indicating the vocabulary size.)
        pad_token_idx: padding token 的 id (A integer indicating the id of padding token.)
        start_token_idx: start token 的 id (A integer indicating the id of start token.)
        end_token_idx: end token 的 id (A integer indicating the id of end token.)
        token_emb_dim: token 嵌入层的维度 (A integer indicating the dimension of token embedding layer.)
        pretrain_embedding: 预训练词向量路径 (A string indicating the path of pretrained embedding.)
        n_word: 词语数量 (A integer indicating the number of words.)
        n_entity: 实体数量 (A integer indicating the number of entities.)
        pad_word_idx: 词语 padding 的 id (A integer indicating the id of word padding.)
        pad_entity_idx: 实体 padding 的 id (A integer indicating the id of entity padding.)
        num_bases: RGCN 的基数 (A integer indicating the number of bases.)
        kg_emb_dim: 知识图谱嵌入维度 (A integer indicating the dimension of kg embedding.)
        n_heads: Transformer 的头数 (A integer indicating the number of heads.)
        n_layers: Transformer 的层数 (A integer indicating the number of layer.)
        ffn_size: Transformer FFN 层的隐藏层大小 (A integer indicating the size of ffn hidden.)
        dropout: dropout 率 (A float indicating the dropout rate.)
        attention_dropout: attention 层的 dropout 率 (A integer indicating the dropout rate of attention layer.)
        relu_dropout: relu 层的 dropout 率 (A integer indicating the dropout rate of relu layer.)
        learn_positional_embeddings: 是否学习位置嵌入 (A boolean indicating if we learn the positional embedding.)
        embeddings_scale: 是否缩放嵌入 (A boolean indicating if we use the embeddings scale.)
        reduction: 是否使用 reduction (A boolean indicating if we use the reduction.)
        n_positions: 位置数量 (A integer indicating the number of position.)
        response_truncate: 生成回复的最大长度 (A integer indicating the longest length for response generation.)
        use_amp: 是否使用混合精度训练 (A boolean indicating if AMP is enabled.)
        scaler: GradScaler 对象，用于混合精度训练 (GradScaler for AMP.)

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): 超参数字典 (A dictionary record the hyper parameters.)
            device (torch.device): 指定数据和模型存放的设备 (A variable indicating which device to place the data and model.)
            vocab (dict): 词汇表信息字典 (A dictionary record the vocabulary information.)
            side_data (dict): 边数据字典 (A dictionary record the side data.)

        """
        self.device = device
        self.gpu = opt.get("gpu", [-1])  # 获取GPU配置，默认为-1 (CPU)
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = vocab['n_word']
        self.n_entity = vocab['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']

        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)

        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)
        # copy mask
        dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "kgsf", dataset)
        resource = resources[dataset]
        self.dpath = dpath  # 保存 dpath 以便后续使用

        # 初始化 AMP (混合精度训练) 相关组件
        self.use_amp = self.device.type == 'cuda'  # 判断是否使用 CUDA，从而决定是否启用 AMP
        self.scaler = GradScaler(enabled=self.use_amp)  # 初始化 GradScaler，仅在 CUDA 环境下启用
        logger.info(f"[AMP] Mixed precision training is {'enabled' if self.use_amp else 'disabled'}.")

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
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
            logger.info("[Embedding] Loaded pretrained token embeddings.")
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.token_emb_dim ** -0.5)  # 使用正态分布初始化
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)  # padding token 初始化为0
            logger.info("[Embedding] Initialized token embeddings from scratch.")

        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)  # 使用正态分布初始化
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)  # padding word 初始化为0
        logger.info("[Embedding] Initialized word KG embeddings.")
        logger.debug('[Finish init embeddings]')

    def _build_kg_layer(self):
        # 构建知识图谱相关的编码器和注意力机制
        # 实体编码器 (DB Encoder)
        self.entity_encoder = FastRGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 实体自注意力

        # 概念编码器 (Concept Encoder)
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)  # 词语自注意力

        # 门控机制 (Gate Mechanism)
        self.gate_layer = GateLayer(self.kg_emb_dim)

        logger.debug('[Finish build kg layer]')

    def _build_infomax_layer(self):
        # 构建 Infomax 相关的层，用于预训练
        self.infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 注意：这里原代码的 bias 是一个 Linear 层，通常 bias 是一个向量
        self.infomax_loss = nn.MSELoss(reduction='sum')  # Infomax 损失函数

        logger.debug('[Finish build infomax layer]')

    def _build_recommendation_layer(self):
        # 构建推荐相关的层
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)  # 推荐偏置项，同上，通常 bias 是向量
        self.rec_loss = nn.CrossEntropyLoss()  # 推荐任务的损失函数

        logger.debug('[Finish build rec layer]')

    def _build_conversation_layer(self):
        # 构建对话生成相关的层
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))  # 注册起始符
        # 对话编码器 (Transformer Encoder)
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
        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

        # 复制机制相关的层
        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)
        # 加载复制掩码，确保只复制词汇表中的特定token
        copy_mask_path = os.path.join(self.dpath, "copy_mask.npy")
        if os.path.exists(copy_mask_path):
            self.copy_mask = torch.as_tensor(np.load(copy_mask_path).astype(bool)).to(self.device)
            logger.info(f"[Copy Mechanism] Loaded copy mask from {copy_mask_path}")
        else:
            logger.warning(f"[Copy Mechanism] copy_mask.npy not found at {copy_mask_path}. Disabling copy mask.")
            # 创建一个全为 False 的掩码，或者根据需要调整
            self.copy_mask = torch.zeros(self.vocab_size, dtype=torch.bool).to(self.device)

        # 对话解码器 (Transformer Decoder with KG integration)
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
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)  # 对话生成的损失函数

        logger.debug('[Finish build conv layer]')

    def pretrain_infomax(self, batch):
        """
        Infomax 预训练阶段
        words: (batch_size, word_length) 输入的词语序列
        entity_labels: (batch_size, n_entity) 实体标签，用于计算 Infomax 损失
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            words, entity_labels = batch

            loss_mask = torch.sum(entity_labels)  # 计算有效的标签数量，用于归一化损失
            if loss_mask.item() == 0:  # 如果没有有效标签，则不计算损失
                logger.warning("[Pretrain Infomax] No entity labels in batch, skipping loss computation.")
                return None

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 获取当前批次词语的表示，并处理 padding
            word_representations = word_graph_representations[words]
            word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)

            # 通过自注意力机制聚合词语表示
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
            word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim) 规范化后的词语信息表示

            # 预测实体，计算 Infomax 损失
            # F.linear(input, weight, bias) -> input.matmul(weight.t()) + bias
            info_predict = F.linear(word_info_rep, entity_graph_representations,
                                    self.infomax_bias.bias)  # (bs, #entity)
            loss = self.infomax_loss(info_predict, entity_labels) / loss_mask  # 计算并归一化损失

        # loss 返回后，在训练循环中需要使用 scaler 进行缩放: scaler.scale(loss).backward()
        return loss

    def recommend(self, batch, mode):
        """
        推荐阶段
        context_entities: (batch_size, entity_length) 上下文中的实体
        context_words: (batch_size, word_length) 上下文中的词语
        entities: (batch_size, n_entity) 推荐相关的实体标签 (用于 Infomax 辅助损失)
        movie: (batch_size) 目标推荐的电影/项目
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            context_entities, context_words, entities, movie = batch

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 处理 padding
            entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
            word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

            # 获取上下文实体和词语的表示
            entity_representations = entity_graph_representations[context_entities]
            word_representations = word_graph_representations[context_words]

            # 通过自注意力机制聚合表示
            entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

            # 通过门控机制融合实体和词语表示，得到用户表示
            user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
            # 计算推荐得分
            rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

            # 计算推荐损失
            rec_loss = self.rec_loss(rec_scores, movie)

            # 计算辅助的 Infomax 损失
            info_loss_mask = torch.sum(entities)
            if info_loss_mask.item() == 0:
                info_loss = None  # 如果没有有效实体标签，则不计算此辅助损失
                # 返回一个零损失或者不返回，这里选择返回 None，调用处需要处理
                logger.debug("[Recommend] No entities for auxiliary infomax loss in batch.")
            else:
                word_info_rep = self.infomax_norm(word_attn_rep)  # (bs, dim)
                info_predict = F.linear(word_info_rep, entity_graph_representations,
                                        self.infomax_bias.bias)  # (bs, #entity)
                info_loss = self.infomax_loss(info_predict, entities) / info_loss_mask

        return rec_loss, info_loss, rec_scores

    def freeze_parameters(self):
        # 冻结指定模型的参数，使其在训练中不更新
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.infomax_norm, self.rec_bias]
        logger.info("[Freeze Parameters] Freezing parameters for specified layers.")
        for model_idx, model in enumerate(freeze_models):
            if model is None:
                logger.warning(f"Model at index {model_idx} is None, skipping freeze.")
                continue
            for param_idx, p in enumerate(model.parameters()):
                p.requires_grad = False
            logger.debug(f"Froze parameters for model: {model.__class__.__name__}")

    def _starts(self, batch_size):
        """返回 bsz 个起始符 (Return bsz start tokens.)"""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):
        # 使用 Teacher Forcing 进行解码 (主要用于训练)
        batch_size, seq_len = response.shape
        # 构建解码器输入，将起始符与真实回复拼接 (去掉最后一个token)
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        # 通过对话解码器获取解码结果
        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)

        # 准备用于复制机制的实体和词语隐状态
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, dim)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)  # (bs, seq_len, dim)

        # 拼接并进行线性变换，得到复制机制的隐状态
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, dim)

        # 计算复制概率和生成概率
        # 复制概率仅限于 copy_mask 中指定的词
        copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1, -1)  # (bs, seq_len, vocab_size)
        # 生成概率通过解码器的隐状态直接映射到词汇表
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)

        # 合并复制和生成概率
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1)  # 预测的 token
        return sum_logits, preds

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask):
        # 使用 Greedy Search 进行解码 (主要用于推理/测试)
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()  # 解码器初始输入为起始符
        incr_state = None  # 用于存储解码器每一步的状态，实现增量解码
        logits_list = []  # 存储每一步的 logits

        for _ in range(self.response_truncate):  # 最多生成 response_truncate 长度的回复
            # 单步解码
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim) 只取最后一个时间步的输出

            # 准备复制机制的隐状态
            db_latent = entity_emb_attn.unsqueeze(1)  # (bs, 1, dim)
            concept_latent = word_emb_attn.unsqueeze(1)  # (bs, 1, dim)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))  # (bs, 1, dim)

            # 计算复制和生成概率
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1, -1)  # (bs, 1, vocab_size)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, 1, vocab_size)
            sum_logits = copy_logits + gen_logits  # (bs, 1, vocab_size)

            preds = sum_logits.argmax(dim=-1).long()  # (bs, 1) 贪心选择概率最大的 token
            logits_list.append(sum_logits)  # 存储 logits
            inputs = torch.cat((inputs, preds), dim=1)  # 将预测的 token 加入到下一次的输入

            # 检查是否所有批次的样本都已生成结束符
            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break

        logits = torch.cat(logits_list, dim=1)  # (bs, seq_len, vocab_size)
        return logits, inputs

    def _decode_beam_search_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                                    word_reps, word_emb_attn, word_mask, beam=4):
        # 使用 Beam Search 进行解码 (用于提升生成质量，但计算量较大)
        # 注意：此处的 Beam Search 实现较为复杂，且在混合精度下可能需要更仔细的调优
        batch_size = token_encoding[0].shape[0]

        # 初始化 beam
        # sequences 存储每个样本的 beam 个候选序列: [sequence_tokens, sequence_logits, sequence_prob]
        sequences = [[(self._starts(1).squeeze().tolist(), [], 1.0)] for _ in range(batch_size)]

        for _ in range(self.response_truncate):

            # 准备当前步的输入和状态
            current_inputs_list = []
            current_token_enc_list_0 = []  # Storing first part of token_encoding tuple
            current_token_enc_list_1 = []  # Storing second part of token_encoding tuple
            current_ent_reps_list = []
            current_ent_attn_list = []
            current_ent_mask_list = []
            current_word_reps_list = []
            current_word_attn_list = []
            current_word_mask_list = []

            # active_beams_indices = [] # 记录哪些 beam 是活跃的 (尚未结束)
            # This list will store tuples of (original_batch_idx, beam_data_idx_in_sequences)
            active_beam_references = []

            for i in range(batch_size):  # 遍历 batch 中的每个样本
                for k_idx, (tokens, _, prob) in enumerate(sequences[i]):  # 遍历当前样本的 beam 个候选
                    if tokens and tokens[-1] == self.end_token_idx and len(tokens) > 1:  # 如果已经结束，则不再扩展
                        # For finished sequences, we just carry them over.
                        # We need a way to add them to all_candidates_next_step[i] later or handle them separately.
                        # For now, let's assume they will be handled when sorting candidates.
                        continue  # Skip expansion for finished beams

                    active_beam_references.append((i, k_idx))  # Store reference to this active beam

                    current_inputs_list.append(torch.tensor(tokens, device=self.device).unsqueeze(0))
                    current_token_enc_list_0.append(token_encoding[0][i:i + 1])
                    current_token_enc_list_1.append(token_encoding[1][i:i + 1])
                    current_ent_reps_list.append(entity_reps[i:i + 1])
                    current_ent_attn_list.append(entity_emb_attn[i:i + 1])
                    current_ent_mask_list.append(entity_mask[i:i + 1])
                    current_word_reps_list.append(word_reps[i:i + 1])
                    current_word_attn_list.append(word_emb_attn[i:i + 1])
                    current_word_mask_list.append(word_mask[i:i + 1])

            if not current_inputs_list:  # 如果所有 beam 都已结束 (no active beams left)
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

            dialog_latent, _ = self.conv_decoder(inputs_batched, token_encoding_batched,
                                                 word_reps_batched, word_mask_batched,
                                                 entity_reps_batched, entity_mask_batched,
                                                 None)  # incr_state 简化为 None for non-incremental beam search step
            dialog_latent = dialog_latent[:, -1:, :]  # Get the last time step

            db_latent = entity_emb_attn_batched.unsqueeze(1)
            concept_latent = word_emb_attn_batched.unsqueeze(1)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))
            copy_logits = self.copy_output(copy_latent) * self.copy_mask.reshape(1, 1, -1)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            sum_logits = copy_logits + gen_logits  # (num_active_beams, 1, vocab_size)

            log_probs = F.log_softmax(sum_logits.squeeze(1), dim=-1)  # (num_active_beams, vocab_size)
            top_log_probs, top_indices = log_probs.topk(beam, dim=-1)  # (num_active_beams, beam)

            # Temporary storage for the next step's beams, before pruning
            temp_sequences = [[] for _ in range(batch_size)]

            active_beam_counter = 0
            for i in range(batch_size):  # Iterate through original batch items
                # Add already finished sequences for this batch item
                for k_idx, (tokens, stored_logits, prob) in enumerate(sequences[i]):
                    if tokens and tokens[-1] == self.end_token_idx and len(tokens) > 1:
                        temp_sequences[i].append((list(tokens), stored_logits, prob))

                # Add expanded sequences for this batch item
                for (orig_batch_idx, beam_data_idx) in active_beam_references:
                    if orig_batch_idx == i:  # If this active beam belongs to the current batch item
                        tokens, _, current_prob = sequences[orig_batch_idx][beam_data_idx]
                        for j in range(beam):  # For each of the `beam` expansions from this active beam
                            new_token = top_indices[active_beam_counter, j].item()
                            new_log_prob_token = top_log_probs[active_beam_counter, j].item()

                            new_tokens_list = list(tokens) + [new_token]
                            new_total_prob = current_prob + new_log_prob_token  # Add log probabilities
                            # Storing logits for beam search is complex; often only probs/scores are kept.
                            # Original code stored `logit_tmp`. For simplicity, we'll store empty list for logits here.
                            temp_sequences[i].append((new_tokens_list, [], new_total_prob))
                        active_beam_counter += 1

            # Prune beams for each batch item
            for i in range(batch_size):
                if temp_sequences[i]:
                    ordered = sorted(temp_sequences[i], key=lambda tup: tup[2], reverse=True)
                    sequences[i] = ordered[:beam]
                elif not sequences[
                    i]:  # If temp_sequences[i] is empty and sequences[i] was also empty (e.g. first step failed)
                    logger.warning(
                        f"Beam search for item {i} resulted in no valid candidates. Re-initializing with start token.")
                    sequences[i] = [(self._starts(1).squeeze().tolist(), [], -float('inf'))]  # Low probability

            # Check if all batch item's top beams have finished
            all_finished_flag = True
            for i in range(batch_size):
                if not sequences[i] or not sequences[i][0][0] or sequences[i][0][0][-1] != self.end_token_idx:
                    all_finished_flag = False
                    break
            if all_finished_flag:
                break

        # Select the best sequence for each batch item
        final_preds_list = []
        for i in range(batch_size):
            if sequences[i] and sequences[i][0][0]:  # Check if there's a valid best sequence
                best_sequence_tokens = sequences[i][0][0]
                # Ensure the sequence ends with EOS if it reached max length but didn't naturally end
                if best_sequence_tokens[-1] != self.end_token_idx and len(
                        best_sequence_tokens) == self.response_truncate:
                    best_sequence_tokens.append(self.end_token_idx)
                final_preds_list.append(torch.tensor(best_sequence_tokens, device=self.device))
            else:  # Fallback if no valid sequence was found (should be rare)
                logger.warning(f"Beam search for item {i} produced no valid best sequence. Using EOS.")
                final_preds_list.append(torch.tensor([self.end_token_idx], device=self.device))

        # The original function expects (logits, inputs).
        # Returning actual logits from beam search is non-trivial if they weren't stored per step.
        # We return None for logits and the list of predicted token tensors.
        return None, final_preds_list

    def converse(self, batch, mode):
        """
        对话阶段
        context_tokens: (batch_size, context_length) 上下文 token
        context_entities: (batch_size, entity_length) 上下文实体
        context_words: (batch_size, word_length) 上下文词语
        response: (batch_size, response_length) 目标回复 (训练时)
        """
        # 使用 autocast 进行混合精度计算
        with autocast(enabled=self.use_amp):
            context_tokens, context_entities, context_words, response_or_none = batch  # response might be None in test mode

            # 获取实体图和词图的表示
            entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
            word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

            # 处理 padding
            entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
            word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, seq_len)

            # 获取上下文实体和词语的表示
            entity_representations = entity_graph_representations[context_entities]
            word_representations = word_graph_representations[context_words]

            # 通过自注意力机制聚合表示
            entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
            word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

            # 对话编码器对上下文 token 进行编码
            tokens_encoding = self.conv_encoder(context_tokens)

            # 准备用于解码器的知识图谱相关嵌入
            conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)
            conv_word_emb = self.conv_word_attn_norm(word_attn_rep)
            conv_entity_reps = self.conv_entity_norm(entity_representations)
            conv_word_reps = self.conv_word_norm(word_representations)

            if mode != 'test':
                # 训练或验证模式，response_or_none 应该是实际的 response tensor
                response = response_or_none
                logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                            entity_padding_mask,
                                                            conv_word_reps, conv_word_emb, word_padding_mask,
                                                            response)

                logits_flat = logits.reshape(-1, logits.shape[-1])
                response_flat = response.reshape(-1)
                loss = self.conv_loss(logits_flat, response_flat)
                return loss, preds
            else:  # 测试模式
                # Beam search is preferred for testing if implemented and stable
                # _, preds = self._decode_beam_search_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                #                                           entity_padding_mask,
                #                                           conv_word_reps, conv_word_emb, word_padding_mask,
                #                                           beam=4) # Example beam size
                # Sticking to greedy for now as per original structure for 'test' in converse
                _, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                       entity_padding_mask,
                                                       conv_word_reps, conv_word_emb, word_padding_mask)
                return preds

    def forward(self, batch, stage, mode):
        """
        模型的前向传播入口
        batch: 当前批次的数据
        stage: 当前阶段 ('pretrain', 'rec', 'conv')
        mode: 当前模式 ('train', 'val', 'test')
        """
        if len(self.gpu) >= 2 and self.device.type == 'cuda':
            current_cuda_device = torch.cuda.current_device()
            self.entity_edge_idx = self.entity_edge_idx.to(current_cuda_device)
            self.entity_edge_type = self.entity_edge_type.to(current_cuda_device)
            self.word_edges = self.word_edges.to(current_cuda_device)

            copy_mask_path = os.path.join(self.dpath, "copy_mask.npy")
            if hasattr(self, 'copy_mask') and self.copy_mask is not None:
                self.copy_mask = self.copy_mask.to(current_cuda_device)
            elif os.path.exists(copy_mask_path):
                self.copy_mask = torch.as_tensor(np.load(copy_mask_path).astype(bool)).to(current_cuda_device)

        if stage == "pretrain":
            loss = self.pretrain_infomax(batch)
            # 预训练阶段，通常只返回一个损失值
            # 调用者 (system.step) 会处理这个损失
            return loss
        elif stage == "rec":
            # self.recommend 返回: rec_loss, info_loss, rec_scores
            rec_loss, info_loss, rec_scores = self.recommend(batch, mode)
            # 在推荐阶段，KGSFSystem 的 step 方法期望 (rec_loss, info_loss, rec_predict)
            # rec_predict 对应 rec_scores
            return rec_loss, info_loss, rec_scores

        elif stage == "conv":
            if mode != 'test':
                # self.converse 在非测试模式下返回 (loss, preds)
                loss, preds = self.converse(batch, mode)
                # KGSFSystem 的 step 方法在非测试对话阶段期望 (loss, preds)
                return loss, preds
            else:
                # self.converse 在测试模式下返回 preds
                preds = self.converse(batch, mode)
                # KGSFSystem 的 step 方法在测试对话阶段期望 preds
                return preds
        else:
            logger.error(f"Unknown stage: {stage}")
            raise ValueError(f"Unknown stage: {stage}")
