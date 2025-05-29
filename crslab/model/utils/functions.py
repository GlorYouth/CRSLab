# -*- encoding: utf-8 -*-
# @Time    :   2020/11/26
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2020/11/16
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import torch
from typing import Tuple, Union, Any, Literal

# 定义输入 edge_data 的可能类型，Any 表示接受多种类型，可以根据实际情况具体化
# 例如: import numpy as np; InputEdgeDataType = Union[List[List[int]], List[Tuple[int, ...]], np.ndarray, torch.Tensor]
InputEdgeDataType = Any  # 保持与原始代码相似的灵活性

# 为图格式类型定义字面量类型，增强类型检查
GraphFormat = Literal['RGCN', 'GCN']

def edge_to_pyg_format(
    edge_data: InputEdgeDataType,
    graph_format_type: GraphFormat = 'RGCN'  # 将参数 'type' 重命名
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    将边数据转换为与 PyTorch Geometric (PyG) 兼容的格式。

    此函数接收边数据（可以是列表的列表/元组、NumPy 数组或 PyTorch 张量），
    并将其转换为标准的 PyG 边索引格式。对于 'RGCN' 类型，它还会提取边的类型。

    参数:
        edge_data: 输入的边数据。
            - 对于 'RGCN': 期望数据可以被转换为一个张量，其中每一行是
              [source_node, target_node, edge_type]。
              示例: `[[0, 1, 0], [1, 2, 1]]`
            - 对于 'GCN': 期望数据是一个 [source_node, target_node] 对的列表，
              或者一个可以被转换为形状为 [num_edges, 2] 的张量的结构。
              示例: `[[0, 1], [1, 2]]`
        graph_format_type: 目标图的格式。必须是 'RGCN' 或 'GCN'。
                           默认为 'RGCN'。

    返回:
        - 如果 `graph_format_type` 是 'RGCN':
            一个元组 `(edge_idx, edge_type)`:
            - `edge_idx` (torch.Tensor): 边索引张量，形状为 [2, num_edges] (dtype=torch.long)。
            - `edge_type` (torch.Tensor): 边类型张量，形状为 [num_edges] (dtype=torch.long)。
        - 如果 `graph_format_type` 是 'GCN':
            `edge_idx` (torch.Tensor): 边索引张量，形状为 [2, num_edges] (dtype=torch.long)。

    可能引发的异常:
        ValueError: 如果 `edge_data` 为空（且不是张量）或其结构不符合
                    指定的 `graph_format_type` 的预期。
        NotImplementedError: 如果 `graph_format_type` 不是 'RGCN' 或 'GCN'。
    """
    # 处理空的非张量输入 (例如空列表)
    if not isinstance(edge_data, torch.Tensor) and not edge_data:
        if graph_format_type == 'RGCN':
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
        elif graph_format_type == 'GCN':
            return torch.empty((2, 0), dtype=torch.long)
        # 如果 graph_format_type 无效，则会落到函数末尾的 NotImplementedError

    if graph_format_type == 'RGCN':
        try:
            # 将输入数据转换为张量。这里假设 edge_data 结构良好
            # (例如，元素为数字的列表的列表、numpy 数组或已经是张量)。
            edge_sets_tensor = torch.as_tensor(edge_data, dtype=torch.long)
        except Exception as e:
            raise ValueError(f"为 RGCN 转换 edge_data 到张量失败: {e}")

        # 处理输入为空列表/元组转换而来的空张量，或本身就是空张量
        if edge_sets_tensor.numel() == 0:
             return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        # 校验张量形状
        if edge_sets_tensor.ndim != 2 or edge_sets_tensor.shape[1] < 3:
            raise ValueError(
                f"对于 'RGCN' 格式, edge_data 必须能转换为一个至少包含3列的2D张量 "
                f"(源节点, 目标节点, 边类型)。实际形状: {edge_sets_tensor.shape}"
            )
        edge_idx = edge_sets_tensor[:, :2].t()  # 提取源节点和目标节点，并转置
        edge_type = edge_sets_tensor[:, 2]      # 提取边类型
        return edge_idx, edge_type
    elif graph_format_type == 'GCN':
        # 如果输入已经是 PyTorch 张量
        if isinstance(edge_data, torch.Tensor):
            edge_tensor = edge_data.to(torch.long) # 确保数据类型
            if edge_tensor.numel() == 0: # 处理空张量
                 return torch.empty((2,0), dtype=torch.long)

            if edge_tensor.ndim == 2:
                if edge_tensor.shape[0] == 2:  # 已经是 [2, num_edges] 格式
                    return edge_tensor
                elif edge_tensor.shape[1] == 2:  # 是 [num_edges, 2] 格式，需要转置
                    return edge_tensor.t()
            # 如果形状不符合上述任一情况
            raise ValueError(
                f"对于 'GCN' 格式, 如果 'edge_data' 是张量, 其形状必须是 [N, 2] 或 [2, N]。 "
                f"实际形状: {edge_tensor.shape}"
            )
        else: # 假定输入是类似列表的结构，包含多个节点对 (原始GCN逻辑)
            try:
                # 此处处理 [[s1, t1], [s2, t2]] 这样的列表的列表/元组
                # 空的 edge_data 在函数开头已处理。
                source_nodes = [co[0] for co in edge_data]
                target_nodes = [co[1] for co in edge_data]
                return torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            except (IndexError, TypeError) as e: # 捕获列表解析中可能发生的错误
                raise ValueError(
                    "对于 'GCN' 格式的类列表输入, 'edge_data' 必须是可迭代的 "
                    f"[源节点, 目标节点] 对。错误: {e}"
                )
    else:
        # 修正了 f-string 的使用
        raise NotImplementedError(f"图格式类型 '{graph_format_type}' 尚未实现。")


def sort_for_packed_sequence(lengths: torch.Tensor):
    """
    :param lengths: 1D array of lengths
    :return: sorted_lengths (lengths in descending order), sorted_idx (indices to sort), rev_idx (indices to retrieve original order)

    """
    sorted_idx = torch.argsort(lengths, descending=True)  # idx to sort by length
    rev_idx = torch.argsort(sorted_idx)  # idx to retrieve original order
    sorted_lengths = lengths[sorted_idx]

    return sorted_lengths, sorted_idx, rev_idx
