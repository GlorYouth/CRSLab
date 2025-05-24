# -*- coding: utf-8 -*-
# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import json  # 导入json用于加载项目文本
import os  # 导入os用于路径检查
from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt


class KGSFDataLoader(BaseDataLoader):
    """Dataloader for model KGSF.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.
        - ``'word_truncate'``: the maximum length of mentioned words in context.
        - ``'item_text_truncate'``: the maximum length of item's text description. (新增)
        - ``'item_texts_path'``: path to the json file mapping item_id to its text description.
                                  (新增, e.g., "data/dataset/redial/item_texts.json").
                                  The JSON file should be a dictionary mapping item_id (str or int) to its text description (str).


        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``
        - ``'pad_word'``
        - ``'unk'`` (用于处理文本描述中未登录词)
        - ``'tokenizer'`` (可选，一个可以将文本字符串转换为token ID列表的函数, e.g., from Hugging Face or a custom one)


        the above values specify the id of needed special token.

        - ``'n_entity'``: the number of entities in the entity KG of dataset.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model. (e.g., an instance of KGSFDataset)
                     Expected to have an attribute like `dataset.dataset_path` pointing to the main data file
                     (e.g., 'data/dataset/redial/nltk/train_data.json') for inferring `item_texts_path`.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.n_entity = vocab['n_entity']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.unk_token_idx = vocab.get('unk', self.pad_token_idx)  # 如果没有unk，则用pad代替

        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)

        # 新增：项目文本描述相关初始化
        self.item_text_truncate = opt.get("item_text_truncate", 50)
        default_item_texts_path = None
        # 尝试从dataset对象获取路径信息以推断 item_texts.json 的默认路径
        if hasattr(dataset, 'dataset_path') and dataset.dataset_path:
            # 假设 item_texts.json 与数据集主文件在同一目录或其父目录
            # 例如, 如果 dataset.dataset_path 是 'data/dataset/redial/nltk/train_data.json',
            # dataset_dir 将是 'data/dataset/redial/nltk/'
            # potential_path 将是 'data/dataset/redial/nltk/item_texts.json'
            dataset_dir = os.path.dirname(dataset.dataset_path)
            potential_path = os.path.join(dataset_dir, "item_texts.json")
            if os.path.exists(potential_path):
                default_item_texts_path = potential_path
            else:
                # 尝试在父目录查找
                parent_dir_potential_path = os.path.join(os.path.dirname(dataset_dir), "item_texts.json")
                if os.path.exists(parent_dir_potential_path):
                    default_item_texts_path = parent_dir_potential_path

        item_texts_path = opt.get("item_texts_path", default_item_texts_path)
        self.item_id2text = self._load_item_texts(item_texts_path)

        # 获取或定义分词器
        # 优先使用 vocab 中提供的 tokenizer，其次是 opt 中的，最后是默认的简单分词
        if 'tokenizer' in vocab and callable(vocab['tokenizer']):
            self.tokenizer = vocab['tokenizer']
            print("使用 vocab 中提供的分词器处理项目文本。")
        elif opt.get('tokenizer_path'):  # 例如，如果配置了Hugging Face分词器路径
            from transformers import AutoTokenizer  # 移到 try 内部以避免不必要的导入
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(opt.get('tokenizer_path'))
                # 确保分词结果是ID列表
                self.tokenizer = lambda text: hf_tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                                                  max_length=self.item_text_truncate)
                print(f"成功加载Hugging Face分词器从 {opt.get('tokenizer_path')}")
            except Exception as e:
                print(f"错误: 加载Hugging Face分词器失败 {opt.get('tokenizer_path')}: {e}. 将使用默认分词器。")
                # Fallback tokenizer
                self.tokenizer = lambda text: [self.vocab.get(tok, self.unk_token_idx) for tok in
                                               str(text).split()[:self.item_text_truncate]]
        else:
            self.tokenizer = lambda text: [self.vocab.get(tok, self.unk_token_idx) for tok in
                                           str(text).split()[:self.item_text_truncate]]
            print("警告: 使用默认的基于空格和词汇表映射的分词器处理项目文本。")

    def _load_item_texts(self, item_texts_path):
        """
        加载项目ID到其文本描述的映射。
        期望的JSON格式: {"item_id_str_or_int": "description_text_str", ...}
        """
        item_id2text = {}
        if item_texts_path and os.path.exists(item_texts_path):
            try:
                with open(item_texts_path, 'r', encoding='utf-8') as f:
                    item_id2text_raw = json.load(f)
                    for k, v in item_id2text_raw.items():
                        try:
                            item_id2text[int(k)] = str(v)  # 确保描述是字符串, key是整数
                        except ValueError:
                            # 如果键不能直接转为int，尝试保持原样（可能已经是处理过的ID）
                            item_id2text[k] = str(v)
                    print(f"成功从 {item_texts_path} 加载 {len(item_id2text)} 条项目文本描述。")
            except Exception as e:
                print(f"错误：加载项目文本描述文件 '{item_texts_path}' 失败: {e}")
                # item_id2text保持为空字典，后续处理会用空字符串代替缺失描述
        else:
            print(f"警告: 未找到项目文本描述文件路径 '{item_texts_path}'。所有项目文本描述将视为空。")
            # item_id2text保持为空字典
        return item_id2text

    def _process_item_text(self, item_id):
        """
        处理单个项目的文本描述。
        将其转换为token_ids序列，并进行padding/truncation。
        """
        # 确保 item_id 是可以在 item_id2text 中查找的类型 (通常是整数)
        try:
            lookup_id = int(item_id)
        except (ValueError, TypeError):
            lookup_id = item_id  # 如果不能转为int，则按原样查找 (例如，如果ID本身就是字符串)

        raw_text = self.item_id2text.get(lookup_id, "")
        if not raw_text:
            token_ids = []
        else:
            try:
                token_ids = self.tokenizer(raw_text)
                if not isinstance(token_ids, list) or (token_ids and not isinstance(token_ids[0], int)):
                    print(
                        f"警告: 分词器未返回预期的整数ID列表 (item_id: {item_id}, text: '{raw_text[:50]}...'). 将使用空列表。")
                    token_ids = []
            except Exception as e:
                print(f"错误: 分词项目文本时出错 (item_id: {item_id}, text: '{raw_text[:50]}...'): {e}. 将使用空列表。")
                token_ids = []

        processed_token_ids = token_ids[:self.item_text_truncate]
        padding_len = self.item_text_truncate - len(processed_token_ids)
        processed_token_ids += [self.pad_token_idx] * padding_len

        return processed_token_ids

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

        return (padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity))

    def rec_process_fn(self):
        """
        预处理推荐数据，为每个推荐项目创建一个单独的数据点。
        期望 `conv_dict['items']` 包含数字形式的项目ID。
        """
        augment_dataset = []
        for conv_dict in tqdm(self.dataset, desc="[Process rec data]"):
            # 通常，推荐行为发生在推荐者（Recommender）的回合
            if conv_dict['role'] == 'Recommender':
                # `items` 字段应包含该回合推荐的项目的数字ID列表
                # 例如，如果原始数据中电影用@12345表示，预处理后这里应该是[12345]
                if 'items' in conv_dict and isinstance(conv_dict['items'], list):
                    for movie_id in conv_dict['items']:
                        augment_conv_dict = deepcopy(conv_dict)
                        augment_conv_dict['item'] = movie_id  # 'item' 字段存储单个目标推荐项目ID
                        augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        batch_item = []
        batch_item_texts = []  # 新增：用于存储项目文本描述

        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

            # conv_dict['item'] 是由 rec_process_fn 添加的单个推荐项目ID
            current_item_id = conv_dict['item']
            batch_item.append(current_item_id)

            processed_item_text_ids = self._process_item_text(current_item_id)
            batch_item_texts.append(processed_item_text_ids)

        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                torch.tensor(batch_item, dtype=torch.long),
                padded_tensor(batch_item_texts, self.pad_token_idx, pad_tail=True)
                )

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []

        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_response, self.pad_token_idx))

    def policy_batchify(self, *args, **kwargs):
        pass

