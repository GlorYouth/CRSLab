# -*- coding: utf-8 -*-
# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/4
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

r"""
KGSF
====
Knowledge-Enhanced Graph-Based Coherent Semantic Fusion for Conversational Recommender System
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionSeq, ContextFusionGate
from crslab.model.utils.modules.transformer import TransformerEncoder, TransformerDecoder

try:
    from crslab.config import MODEL_PATH
except ImportError:
    MODEL_PATH = os.environ.get("CRSLAB_MODEL_PATH", os.path.expanduser("~/.crslab/model"))
    print(f"警告: 无法从 crslab.config 导入 MODEL_PATH，使用默认值或环境变量: {MODEL_PATH}")

try:
    from .resources import resources
except ImportError:
    resources = None
    print("警告: 无法导入同级目录的 resources.py。")


class KGSFModel(BaseModel):
    """KGSF Model Entity
    """

    def __init__(self, opt, device, vocab, side_data):
        """
        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to use.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.
        """
        self.device = device
        self.gpu = opt.get("gpu", [-1])

        self.n_entity = vocab['n_entity']
        self.n_relation = vocab['n_relation']
        self.n_word = vocab['n_word']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab.get('pad_entity', self.pad_token_idx)
        self.pad_word_idx = vocab.get('pad_word', self.pad_token_idx)

        self.item_ids = side_data.get('item_entity_ids', None)

        self.dim = opt['dim']
        self.kg_emb_dim = opt.get('kg_emb_dim', self.dim)
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        # self.reduction = opt['reduction'] # reduction 参数对 TransformerEncoder 有效，对 Decoder 无效
        self.n_positions = opt['n_positions']
        self.response_truncate = opt['response_truncate']
        self.num_bases = opt.get('num_bases', 0)

        self.item_text_emb_dim = opt.get('item_text_emb_dim', self.kg_emb_dim)
        self.item_text_gru_hidden_size = opt.get('item_text_gru_hidden_size', self.dim)
        self.item_text_gru_num_layers = opt.get('item_text_gru_num_layers', 1)

        if 'kg_edge' in side_data and side_data['kg_edge'] is not None:
            edge_index_cpu, edge_type_cpu = edge_to_pyg_format(side_data['kg_edge'], 'RGCN')
            self.edge_index = edge_index_cpu.to(self.device)
            self.edge_type = edge_type_cpu.to(self.device)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_type = torch.empty((0,), dtype=torch.long, device=self.device)
            print("警告: side_data 中未找到 'kg_edge' 或 'kg_edge' 为 None。图网络可能无法正常工作。")

        if self.num_bases == 0:
            self.num_bases = None

        dataset_name = opt.get('dataset', 'unknown_dataset')
        self.dpath = os.path.join(MODEL_PATH, "kgsf", dataset_name)
        os.makedirs(self.dpath, exist_ok=True)

        resource = None
        if resources and dataset_name in resources:
            resource = resources[dataset_name]
        elif resources:
            print(f"警告: 数据集 '{dataset_name}' 未在 resources 中找到。")

        if resource is not None:
            super(KGSFModel, self).__init__(opt, device, dpath=self.dpath, resource=resource)
        else:
            print(f"警告: 未找到数据集 '{dataset_name}' 的有效resource。BaseModel可能未完全初始化。")
            super(KGSFModel, self).__init__(opt, device)

    def build_model(self):
        """build model"""
        self._init_embeddings()
        self._init_kg_network()
        self._init_text_network()
        self._init_item_text_encoder()
        self._init_conv_network()

        copy_mask_path = os.path.join(self.dpath, "copy_mask.npy")
        if os.path.exists(copy_mask_path):
            self.copy_mask = torch.as_tensor(np.load(copy_mask_path).astype(bool)).to(self.device)
            print(f"成功加载 copy_mask 从 {copy_mask_path}")
        else:
            self.copy_mask = None
            print(f"警告: copy_mask.npy 未在 {copy_mask_path} 找到。拷贝机制可能无法正常工作。")

    def _init_embeddings(self):
        self.entity_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim)
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        if self.pad_entity_idx < self.n_entity:
            nn.init.constant_(self.entity_embedding.weight[self.pad_entity_idx], 0)
        self.item_embedding = self.entity_embedding

        self.word_embedding = nn.Embedding(self.n_word, self.dim, padding_idx=self.pad_token_idx)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=self.dim ** -0.5)
        nn.init.constant_(self.word_embedding.weight[self.pad_token_idx], 0)

    def _init_kg_network(self):
        if hasattr(self, 'n_relation') and self.n_relation > 0:
            self.kg_encoder = RGCNConv(self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        else:
            self.kg_encoder = None
            print("警告: n_relation 未正确设置或为0。RGCN编码器未初始化。")
        self.kg_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

    def _init_text_network(self):
        self.token_emb_to_feature = nn.Linear(self.dim, self.dim)
        self.word_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.n_word,
            embedding=self.word_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.opt.get('reduction', True),  # reduction 参数应从 opt 获取并传递给 Encoder
            n_positions=self.n_positions,
        )
        self.word_attn = SelfAttentionSeq(self.dim, self.dim)

        if self.dim != self.kg_emb_dim:
            self.word_vec_to_kg_dim_proj = nn.Linear(self.dim, self.kg_emb_dim)
            print(f"已初始化 word_vec_to_kg_dim_proj: Linear({self.dim}, {self.kg_emb_dim})")
        else:
            self.word_vec_to_kg_dim_proj = nn.Identity()

    def _init_item_text_encoder(self):
        """初始化项目文本描述的编码器"""
        self.item_text_encoder = nn.GRU(
            input_size=self.dim,
            hidden_size=self.item_text_gru_hidden_size,
            num_layers=self.item_text_gru_num_layers,
            batch_first=True,
            bidirectional=False
        )

        if self.item_text_gru_hidden_size != self.item_text_emb_dim:
            self.item_text_proj = nn.Linear(self.item_text_gru_hidden_size, self.item_text_emb_dim)
        else:
            self.item_text_proj = nn.Identity()

        if self.item_text_emb_dim != self.kg_emb_dim:
            print(f"警告: 项目文本最终嵌入维度 item_text_emb_dim ({self.item_text_emb_dim}) "
                  f"与 KG嵌入维度 kg_emb_dim ({self.kg_emb_dim}) 不一致。 "
                  f"在 _rec_logic 中融合时可能需要额外的投影或会导致维度错误。 "
                  f"建议通过opt配置使 item_text_emb_dim 等于 kg_emb_dim。")

    def _init_conv_network(self):
        self.context_gate = ContextFusionGate(
            word_context_dim=self.dim,
            entity_context_dim=self.kg_emb_dim,
            decoder_state_dim=self.dim,
            output_dim=self.dim
        )
        self.recs_fusion = nn.Linear(self.dim + self.kg_emb_dim, self.dim)
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity, bias=True)

        self.decoder = TransformerDecoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.n_word,
            embedding=self.word_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            # reduction=self.reduction, # 已移除此无效参数
            n_positions=self.n_positions,
        )

    def _get_kg_representation(self, entity_ids=None):
        """获取实体表示，如果提供了entity_ids，则获取特定实体的表示"""
        if self.kg_encoder is not None and self.edge_index.numel() > 0:
            all_entity_embeddings = self.entity_embedding.weight
            if all_entity_embeddings.device != self.edge_index.device:
                all_entity_embeddings = all_entity_embeddings.to(self.edge_index.device)

            entity_reps_matrix = self.kg_encoder(all_entity_embeddings, self.edge_index, self.edge_type)
        else:
            entity_reps_matrix = self.entity_embedding.weight

        if entity_ids is not None:
            return entity_reps_matrix[entity_ids]
        return entity_reps_matrix

    def _encode_item_texts(self, item_texts_ids):
        """
        对批量项目文本ID序列进行编码。
        Args:
            item_texts_ids (torch.Tensor): (batch_size, item_text_truncate)
        Returns:
            torch.Tensor: (batch_size, self.item_text_emb_dim)
        """
        item_text_token_embeds = self.word_embedding(item_texts_ids)
        _, last_hidden = self.item_text_encoder(item_text_token_embeds)

        if self.item_text_gru_num_layers == 1 and not self.item_text_encoder.bidirectional:
            item_text_vec = last_hidden.squeeze(0)
        else:
            item_text_vec = last_hidden[-1]

        projected_item_text_vec = self.item_text_proj(item_text_vec)
        return projected_item_text_vec

    def _converse_logic(self, batch, mode):
        context_tokens = batch['context_tokens']
        all_entity_kg_reps = self._get_kg_representation()
        word_contexts_encoded = self.word_encoder(context_tokens)
        word_context_vec, entity_context_vec = self.extract_context_embedding(
            word_contexts_encoded,
            all_entity_kg_reps,
            batch
        )

        decoder_state_proxy = (word_context_vec + entity_context_vec.to(word_context_vec.device)) * 0.5

        dialog_context_vector = self.context_gate(
            word_context=word_context_vec,
            entity_context=entity_context_vec,
            decoder_state=decoder_state_proxy
        )
        response_context_vector = dialog_context_vector

        if mode == 'test' or mode == 'val':
            token_logits = None
            response = self.generate_response(response_context_vector, batch)
            conv_loss = torch.tensor(0.0).to(self.device)
        else:
            response = None
            token_logits = self.decoder(
                input=batch['response'][:, :-1],
                encoder_output=response_context_vector.unsqueeze(1)
            )
            conv_loss = F.cross_entropy(
                token_logits.reshape(-1, token_logits.size(-1)),
                batch['response'][:, 1:].reshape(-1),
                ignore_index=self.pad_token_idx
            )
        return conv_loss, response

    def _rec_logic(self, batch, mode):
        context_entities, context_words, _, target_item_ids, target_item_texts = batch

        context_entity_embeds = self.entity_embedding(context_entities)
        context_entity_mask = (context_entities == self.pad_entity_idx)
        entity_context_vec = self.kg_attn(context_entity_embeds, context_entity_mask)

        context_word_embeds = self.word_embedding(context_words)
        context_word_mask = (context_words == self.pad_word_idx)
        word_context_vec_for_rec = self.word_attn(context_word_embeds, context_word_mask)

        projected_word_context_vec = self.word_vec_to_kg_dim_proj(word_context_vec_for_rec)
        user_rep_base = (entity_context_vec + projected_word_context_vec) * 0.5

        target_item_text_vec = self._encode_item_texts(target_item_texts)

        final_target_item_text_vec = target_item_text_vec
        if self.item_text_emb_dim != self.kg_emb_dim:
            print(
                f"严重警告: _rec_logic 中 item_text_emb_dim ({self.item_text_emb_dim}) 与 kg_emb_dim ({self.kg_emb_dim}) 不匹配!"
                f"请确保通过opt配置 item_text_emb_dim = kg_emb_dim，或修改模型以包含必要的投影。")

        user_rep_final = user_rep_base + final_target_item_text_vec.to(user_rep_base.device)

        rec_scores = self.rec_bias(user_rep_final)

        rec_loss = F.cross_entropy(rec_scores, target_item_ids)

        info_loss = None

        return rec_loss, info_loss, rec_scores

    def forward(self, batch, stage, mode):
        if len(self.gpu) >= 2:
            current_device = torch.cuda.current_device()
            if self.edge_index is not None and self.edge_index.numel() > 0:
                self.edge_index = self.edge_index.to(current_device)
            if self.edge_type is not None and self.edge_type.numel() > 0:
                self.edge_type = self.edge_type.to(current_device)
            if hasattr(self, 'copy_mask') and self.copy_mask is not None:
                self.copy_mask = self.copy_mask.to(current_device)

        if stage == "pretrain":
            print("Pretrain stage called, but not implemented in this version.")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True if mode == 'train' else False)
            return loss

        elif stage == "rec":
            rec_loss, info_loss, rec_scores = self._rec_logic(batch, mode)
            loss = rec_loss
            if info_loss is not None:
                loss = loss + info_loss
            return loss, rec_scores

        elif stage == "conv":
            conv_loss, preds = self._converse_logic(batch, mode)
            return conv_loss, preds
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def extract_context_embedding(self, word_contexts_encoded, all_entity_kg_reps, batch_dict):
        context_tokens_ids = batch_dict['context_tokens']
        context_tokens_mask = (context_tokens_ids == self.pad_token_idx)
        word_context_reps = self.word_attn(word_contexts_encoded, context_tokens_mask)

        context_entity_ids = batch_dict['context_entities']
        context_entities_representations = all_entity_kg_reps[context_entity_ids]

        context_entities_mask = (context_entity_ids == self.pad_entity_idx)
        entity_context_reps = self.kg_attn(context_entities_representations, context_entities_mask)

        return word_context_reps, entity_context_reps

    def generate_response(self, context_vector, batch):
        bs = context_vector.size(0)
        self.decoder.to(context_vector.device)
        generated_response = self.decoder.generate(
            encoder_output=context_vector.unsqueeze(1),
            max_length=self.response_truncate,
            bos_token_id=self.start_token_idx,
            eos_token_id=self.end_token_idx,
            pad_token_id=self.pad_token_idx,
        )
        return generated_response

