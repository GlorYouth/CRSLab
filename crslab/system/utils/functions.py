# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2020/12/18
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE:
# @Time   : 2021/10/05
# @Author  :   Zhipeng Zhao
# @email   :   oran_official@outlook.com

import torch  # 导入 PyTorch 库


def compute_grad_norm(parameters, norm_type=2.0):
    """
    计算模型参数梯度的范数 (优化版本)。

    :param parameters:
        用于计算梯度范数的模型参数。可以是单个 Tensor 或 Tensor 的可迭代对象。
    :param norm_type:
        p-范数的类型，默认为 2.0 (L2 范数)。也可以是 float('inf')。

    :returns:
        计算得到的梯度范数 (Tensor)。
    """
    # 1. 参数预处理和梯度提取
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # 提取所有有效的梯度，并从计算图中分离它们
    # 使用 .detach() 是好习惯，确保这些操作不影响反向传播图
    grads = [p.grad.detach() for p in parameters if p is not None and p.grad is not None]

    # 如果没有梯度，则返回 0
    if not grads:
        # 尝试获取参数的设备，如果参数列表为空或不含Tensor，则默认为cpu
        # 确保返回的是一个tensor，与有梯度时返回类型一致
        device = torch.device('cpu')
        if parameters and isinstance(parameters[0], torch.Tensor) and hasattr(parameters[0], 'device'):
            device = parameters[0].device
        elif parameters and parameters[0] is not None and hasattr(parameters[0], 'grad') and parameters[
            0].grad is not None:
            device = parameters[0].grad.device  # 备选方案，如果第一个参数没有梯度，但后续有
        return torch.tensor(0., device=device)

    norm_type = float(norm_type)  # 确保 norm_type 是浮点数

    # 获取第一个有效梯度的设备，用于后续张量创建
    # 此时 grads 列表不为空
    device = grads[0].device

    # 2. 处理无穷范数 (Infinity Norm)
    if norm_type == float('inf'):
        # 无穷范数是所有梯度元素绝对值的最大值
        # total_norm = max(g.abs().max() for g in grads) # 这会返回一个0维张量
        # 我们可以迭代计算最大值，以保持张量类型
        current_max_norm = torch.tensor(0.0, device=device)
        for g in grads:
            current_max_norm = torch.maximum(current_max_norm, g.abs().max())
        return current_max_norm

    # 3. 处理 p-范数 (Lp Norm)
    # 初始化 total_norm_accumulator 为一个在正确设备上的 0 值张量
    total_norm_accumulator = torch.tensor(0.0, device=device)

    for g in grads:
        # 计算当前梯度张量 g 的所有元素的 norm_type 次方之和
        # 即 sum(|g_j|^p)
        # 这比 g.norm(norm_type).pow(norm_type) 可能更直接，
        # 因为 g.norm() 内部会计算一次开方 (p 次根)，pow(norm_type) 又会进行 p 次幂。
        # 对于 norm_type = 2，这等价于 torch.sum(g.pow(2))
        if norm_type % 2 == 0 or norm_type < 1:  # 对于偶数次幂或0到1之间的幂，可以不用abs()，但为了通用性保留abs()
            # 更安全的做法是始终使用abs()，除非明确norm_type是偶数且为正
            # 例如，如果norm_type是2，g.abs().pow(2) 和 g.pow(2) 结果一样
            # 如果norm_type是1，g.abs().pow(1) == g.abs().sum()
            total_norm_accumulator += torch.sum(g.abs().pow(norm_type))
        else:  # 对于奇数次幂且norm_type > 1 (如 L3 范数等)
            total_norm_accumulator += torch.sum(g.abs().pow(norm_type))
            # 注：torch.norm(p) 的内部实现会处理符号问题，但这里我们是 (sum |x_i|^p)^(1/p)
            # 所以直接 g.abs().pow(norm_type) 是正确的。

    # 最终范数是 (累加和) ^ (1/norm_type)
    # total_norm_accumulator 现在是 sum_i (sum_j |g_ij|^p)
    final_norm = total_norm_accumulator.pow(1.0 / norm_type)

    return final_norm


def ind2txt(inds, ind2tok, end_token_idx=None, unk_token='unk'):
    sentence = []
    for ind in inds:
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        if end_token_idx and ind == end_token_idx:
            break
        sentence.append(ind2tok.get(ind, unk_token))
    return ' '.join(sentence)

def ind2txt_with_slots(inds,slots,ind2tok, end_token_idx=None, unk_token='unk',slot_token='[ITEM]'):
    sentence = []
    for ind in inds:
        if isinstance(ind, torch.Tensor):
            ind = ind.item()
        if end_token_idx and ind == end_token_idx:
            break
        token = ind2tok.get(ind, unk_token)
        if token == slot_token:
            token = slots[0]
            slots = slots[1:] 
        sentence.append(token)
    return ' '.join(sentence)

def ind2slot(inds,ind2slot):
    return [ ind2slot[ind] for ind in inds]
