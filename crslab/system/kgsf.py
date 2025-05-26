# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import torch
from loguru import logger
# 导入 AMP 相关模块
from torch.cuda.amp import GradScaler

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt, compute_grad_norm


class KGSFSystem(BaseSystem):
    """这是 KGSF 模型的 System 类"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """
        初始化 KGSFSystem

        参数:
            opt (dict): 超参数字典。
            train_dataloader (BaseDataLoader): 训练数据加载器。
            valid_dataloader (BaseDataLoader): 验证数据加载器。
            test_dataloader (BaseDataLoader): 测试数据加载器。
            vocab (dict): 词汇表。
            side_data (dict): 边数据。
            restore_system (bool, optional): 是否在训练后恢复系统。默认为 False。
            interact (bool, optional): 是否与系统交互。默认为 False。
            debug (bool, optional): 是否以调试模式训练。默认为 False。
            tensorboard (bool, optional): 是否使用 tensorboard 监控训练。默认为 False。
        """
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok'] #索引到 token 的映射
        self.end_token_idx = vocab['end'] # 结束 token 的 ID
        self.item_ids = side_data['item_entity_ids'] # 项目实体 ID 列表

        # 优化器和训练周期配置
        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        # AMP (混合精度训练) 相关初始化
        self.use_amp = self.device.type == 'cuda' # 判断是否使用 CUDA
        self.scaler = GradScaler(enabled=self.use_amp, init_scale=2.**12) # 尝试较小的初始缩放因子，例如 2^12 或 2^10 # 初始化 GradScaler
        logger.info(f"[System AMP] 系统层面混合精度训练已{'启用' if self.use_amp else '禁用'}.")

        # 梯度累积相关
        self.update_freq = self.opt.get('update_freq', 1) # 获取梯度累积频率，默认为1
        self._number_grad_accum = 0 # 梯度累积计数器
        logger.info(f"[System Grad Accum] 梯度累积频率设置为: {self.update_freq}")


    def backward(self, loss):
        """
        执行反向传播。如果启用了混合精度，则使用 GradScaler。
        如果配置了梯度累积，则相应地调整损失。

        参数:
            loss (torch.Tensor): 当前迭代计算得到的原始损失值 (应为标量)。
        """
        loss_for_backward = loss
        if self.update_freq > 1:
            # 此计数器用于判断何时实际执行优化器步骤和梯度清零
            # 根据原始结构，在此处更新
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss_for_backward = loss / self.update_freq # 为梯度累积缩放损失

        if self.use_amp:
            self.scaler.scale(loss_for_backward).backward() # 使用 GradScaler 缩放损失并反向传播
        else:
            loss_for_backward.backward() # 标准的反向传播

    def _optimizer_step(self):
        """
        执行优化器步骤和梯度清零。
        如果启用了混合精度，则使用 GradScaler。
        同时计算并记录梯度范数。
        """
        # 梯度范数计算和裁剪应该在 optimizer.step() 之前

        # 当使用 AMP 时，梯度裁剪必须在 scaler.unscale_ 之后和 scaler.step 之前进行
        if self.use_amp:
            self.scaler.unscale_(self.optimizer) # 在裁剪或计算范数前反缩放梯度

        # 从 BaseSystem.init_optim 获取 gradient_clip 和 parameters
        # self.parameters 是在 BaseSystem.init_optim 中设置的
        # self.gradient_clip 也是在 BaseSystem.init_optim 中设置的

        if hasattr(self, 'gradient_clip') and self.gradient_clip > 0:
            # clip_grad_norm_ 返回的是裁剪后的总范数
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), # 或者 self.parameters (两者应指向相同集合)
                self.gradient_clip
            )
            # grad_norm 可能是一个 tensor，确保转为 python float
            grad_norm_item = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm_item))
            self.evaluator.optim_metrics.add(
                'grad clip ratio', # 评估梯度被裁剪的频率/程度
                AverageMetric(float(grad_norm_item > self.gradient_clip)) # 使用 grad_norm_item
            )
        else:
            # 如果不进行梯度裁剪，也计算范数用于监控
            # compute_grad_norm 需要一个参数列表
            # self.parameters 由 BaseSystem.init_optim 初始化，它是一个列表
            grad_norm = compute_grad_norm(self.parameters if hasattr(self, 'parameters') else list(self.model.parameters()))
            grad_norm_item = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm_item))

        # 执行优化器步骤
        if self.use_amp:
            self.scaler.step(self.optimizer) # GradScaler 会处理已反缩放的梯度
            self.scaler.update() # 更新 GradScaler 的缩放因子

        else:
            self.optimizer.step() # 标准的优化器步骤

        self.optimizer.zero_grad() # 参数更新后清零梯度

    def rec_evaluate(self, rec_predict, item_label):
        """推荐任务评估"""
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids] # 只选择 item ID 对应的预测结果
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1) # 取 top 50
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            try:
                item_idx_in_eval_subset = self.item_ids.index(item) # 获取 item 在 item_ids 中的索引
                self.evaluator.rec_evaluate(rec_rank, item_idx_in_eval_subset)
            except ValueError:
                # 如果 item_label 中的 item 不在 self.item_ids (用于评估的子集) 中，则跳过
                # logger.debug(f"评估跳过: 项目 {item} 不在用于评估的项目ID列表中。")
                pass


    def conv_evaluate(self, prediction, response):
        """对话任务评估"""
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx) # 预测转换为文本
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx) # 真实回复转换为文本
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        """
        执行单个训练/评估步骤。

        参数:
            batch (list of torch.Tensor): 当前批次的数据。
            stage (str): 当前阶段 ('pretrain', 'rec', 'conv')。
            mode (str): 当前模式 ('train', 'val', 'test')。
        """
        batch = [ele.to(self.device) for ele in batch] # 数据移至设备

        if stage == 'pretrain':
            # 模型前向传播 (模型内部已使用 autocast)
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                current_loss = info_loss.sum() # 确保损失是标量
                if mode == "train":
                    self.backward(current_loss) # 执行反向传播
                    # 检查是否需要执行优化器步骤
                    if self._number_grad_accum == 0 or self.update_freq == 1:
                        self._optimizer_step()

                scalar_info_loss = current_loss.item() # 用于记录的标量损失
                if self.update_freq > 1 : scalar_info_loss *= self.update_freq # 恢复原始损失大小进行记录
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(scalar_info_loss))

        elif stage == 'rec':
            # 模型前向传播
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)

            # 计算总损失
            if info_loss is not None: # 确保 info_loss 存在
                combined_loss = rec_loss + 0.025 * info_loss
            else:
                combined_loss = rec_loss

            current_loss = combined_loss.sum() # 确保损失是标量

            if mode == "train":
                self.backward(current_loss) # 执行反向传播
                # 检查是否需要执行优化器步骤
                if self._number_grad_accum == 0 or self.update_freq == 1:
                    self._optimizer_step()
            else: # 'val' or 'test'
                self.rec_evaluate(rec_predict, batch[-1]) # 评估推荐结果

            # 记录损失指标
            scalar_rec_loss = rec_loss.sum().item()
            if self.update_freq > 1 and mode == "train": scalar_rec_loss *= self.update_freq
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(scalar_rec_loss))
            if info_loss is not None:
                scalar_info_loss = info_loss.sum().item()
                if self.update_freq > 1 and mode == "train": scalar_info_loss *= self.update_freq
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(scalar_info_loss))

        elif stage == "conv":
            if mode != "test":
                # 模型前向传播
                gen_loss, pred = self.model.forward(batch, stage, mode)
                current_loss = gen_loss.sum() # 确保损失是标量

                if mode == 'train':
                    self.backward(current_loss) # 执行反向传播
                    # 检查是否需要执行优化器步骤
                    if self._number_grad_accum == 0 or self.update_freq == 1:
                        self._optimizer_step()
                else: # 'val'
                    self.conv_evaluate(pred, batch[-1]) # 评估对话结果

                scalar_gen_loss = current_loss.item()
                if self.update_freq > 1 and mode == "train": scalar_gen_loss *= self.update_freq
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(scalar_gen_loss))
                # PPL 计算应使用未调整的损失 (每个 token 的平均损失)
                # 如果 gen_loss 是 batch 内 token 的总和损失，PPLMetric 可能需要平均后的损失
                # 假设 PPLMetric 内部处理了平均
                self.evaluator.gen_metrics.add("ppl", PPLMetric(scalar_gen_loss)) # PPL 通常基于每个token的交叉熵
            else: # 'test'
                # 模型前向传播
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1]) # 评估对话结果
        else:
            logger.error(f"未知的阶段: {stage}")
            raise ValueError(f"未知的阶段: {stage}")

    def pretrain(self):
        """预训练阶段"""
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())
        self._number_grad_accum = 0 # 每个训练阶段开始时重置累积计数器

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[预训练 Epoch {str(epoch)}]')
            # 注意：get_pretrain_data 的 shuffle 参数原为 False，通常训练时应为 True
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=True):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        """训练推荐模块"""
        self.init_optim(self.rec_optim_opt, self.model.parameters())
        self._number_grad_accum = 0 # 重置累积计数器

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[推荐 Epoch {str(epoch)}]')

            logger.info('[训练]')
            # 注意：get_rec_data 的 shuffle 参数原为 False
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=True):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')

            logger.info('[验证]')
            with torch.no_grad(): # 验证和测试时不需要计算梯度
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # 早停逻辑
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    logger.info("触发早停机制。")
                    break

        logger.info('[测试]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            # 加载最佳模型进行测试 (如果实现了保存和加载最佳模型的逻辑)
            # self.load_model(self.opt['MODEL_PATH'] ... )
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        """训练对话模块"""
        # 这部分检查 GPU 可见性的代码似乎与参数初始化有关，但通常直接初始化优化器即可
        # if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
        #     self.model.named_parameters() # 仅迭代参数，无实际作用
        # else:
        #     self.model.named_parameters() # 仅迭代参数，无实际作用
        self.init_optim(self.conv_optim_opt, self.model.parameters())
        self._number_grad_accum = 0 # 重置累积计数器

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[对话 Epoch {str(epoch)}]')

            logger.info('[训练]')
            # 注意：get_conv_data 的 shuffle 参数原为 False
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=True):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')

            logger.info('[验证]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # 对话模块通常基于 PPL 或其他生成指标进行早停，这里未实现

        logger.info('[测试]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        """执行完整的训练流程：预训练 -> 推荐训练 -> 对话训练"""
        if self.pretrain_epoch > 0:
            self.pretrain()
        else:
            logger.info("[跳过预训练阶段]")

        if self.rec_epoch > 0:
            # 在训练推荐器之前，可能需要冻结模型的某些部分 (例如，如果对话模块已预训练)
            # 或者加载预训练的权重（如果适用）
            # self.model.freeze_parameters() # 示例：如果 KGSFModel 有此方法
            self.train_recommender()
        else:
            logger.info("[跳过推荐模块训练阶段]")

        if self.conv_epoch > 0:
            # 在训练对话模块之前，可能需要加载推荐模块训练好的权重
            # 或者解冻/调整模型的某些部分
            self.train_conversation()
        else:
            logger.info("[跳过对话模块训练阶段]")


    def interact(self):
        """与系统进行交互 (当前未实现)"""
        # TODO: 实现交互逻辑
        logger.info("交互模式当前未实现。")
        pass
