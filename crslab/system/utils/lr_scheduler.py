# @Time   : 2020/12/1
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

from abc import abstractmethod, ABC
from typing import Literal

# UPDATE:
# @Time   : 2020/12/14
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com
import math
from abc import abstractmethod, ABC
from typing import Literal
import numpy as np
import torch
from loguru import logger
from torch import optim


class LRScheduler(ABC):
    """
    Class for LR Schedulers.

    Includes some basic functionality by default - setting up the warmup
    scheduler, passing the correct number of steps to train_step, loading and
    saving states.
    Subclasses must implement abstract methods train_step() and valid_step().
    Schedulers should be initialized with lr_scheduler_factory().
    __init__() should not be called directly.
    """

    def __init__(self, optimizer, warmup_steps: int = 0):
        """
        Initialize warmup scheduler. Specific main schedulers should be initialized in
        the subclasses. Do not invoke this method diretly.

        :param optimizer optimizer:
            Optimizer being used for training. May be wrapped in
            fp16_optimizer_wrapper depending on whether fp16 is used.
        :param int warmup_steps:
            Number of training step updates warmup scheduler should take.
        """
        self._number_training_updates = 0
        self.warmup_steps = warmup_steps
        self._init_warmup_scheduler(optimizer)

    def _warmup_lr(self, step):
        """
        Return lr multiplier (on initial lr) for warmup scheduler.
        """
        return float(step) / float(max(1, self.warmup_steps))

    def _init_warmup_scheduler(self, optimizer):
        if self.warmup_steps > 0:
            self.warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, self._warmup_lr)
        else:
            self.warmup_scheduler = None

    def _is_lr_warming_up(self):
        """
        Check if we're warming up the learning rate.
        """
        return (
                hasattr(self, 'warmup_scheduler')
                and self.warmup_scheduler is not None
                and self._number_training_updates <= self.warmup_steps
        )

    def train_step(self):
        """
        Use the number of train steps to adjust the warmup scheduler or the main
        scheduler, depending on where in training we are.

        Override this method to override the behavior for training schedulers.
        """
        self._number_training_updates += 1
        if self._is_lr_warming_up():
            self.warmup_scheduler.step()
        else:
            self.train_adjust()

    def valid_step(self, metric=None):
        if self._is_lr_warming_up():
            # we're not done warming up, so don't start using validation
            # metrics to adjust schedule
            return
        self.valid_adjust(metric)

    @abstractmethod
    def train_adjust(self):
        """
        Use the number of train steps to decide when to adjust LR schedule.

        Override this method to override the behavior for training schedulers.
        """
        pass

    @abstractmethod
    def valid_adjust(self, metric):
        """
        Use the metrics to decide when to adjust LR schedule.

        This uses the loss as the validation metric if present, if not this
        function does nothing. Note that the model must be reporting loss for
        this to work.

        Override this method to override the behavior for validation schedulers.
        """
        pass


class ReduceLROnPlateau(LRScheduler):
    """
    Scheduler that decays by a multiplicative rate when valid loss plateaus.
    """

    def __init__(self, optimizer, mode: Literal["min", "max"]= "min", factor=0.1, patience=10, threshold=0.0001,
                 threshold_mode: Literal["rel", "abs"] = "rel", cooldown=0, min_lr=0, eps=1e-08, warmup_steps=0):
        super(ReduceLROnPlateau, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=mode, factor=factor,
                                                              patience=patience, threshold=threshold,
                                                              threshold_mode=threshold_mode, cooldown=cooldown,
                                                              min_lr=min_lr, eps=eps)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric):
        self.scheduler.step(metric)


class StepLR(LRScheduler):
    """
    Scheduler that decays by a fixed multiplicative rate at each valid step.
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, warmup_steps=0):
        super(StepLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric=None):
        self.scheduler.step()


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps=0):
        super(ConstantLR, self).__init__(optimizer, warmup_steps)

    def train_adjust(self):
        pass

    def valid_adjust(self, metric):
        pass


class InvSqrtLR(LRScheduler):
    """
    Scheduler that decays at an inverse square root rate.
    """

    def __init__(self, optimizer, invsqrt_lr_decay_gamma=-1, last_epoch=-1, warmup_steps=0):
        """
        invsqrt_lr_decay_gamma determines the cycle length of the inverse square root
        scheduler.

        When steps taken == invsqrt_lr_decay_gamma, the lr multiplier is 1
        """
        super(InvSqrtLR, self).__init__(optimizer, warmup_steps)
        self.invsqrt_lr_decay_gamma = invsqrt_lr_decay_gamma
        if invsqrt_lr_decay_gamma <= 0:
            logger.warning(
                '--lr-scheduler invsqrt requires a value for '
                '--invsqrt-lr-decay-gamma. Defaulting to set gamma to '
                '--warmup-updates value for backwards compatibility.'
            )
            self.invsqrt_lr_decay_gamma = self.warmup_steps

        self.decay_factor = np.sqrt(max(1, self.invsqrt_lr_decay_gamma))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._invsqrt_lr, last_epoch)

    def _invsqrt_lr(self, step):
        return self.decay_factor / np.sqrt(max(1, self.invsqrt_lr_decay_gamma + step))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        # this is a training step lr scheduler, nothing to adjust in validation
        pass


class CosineAnnealingLR(LRScheduler):
    """
    Scheduler that decays by a cosine function.
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0):
        """
        training_steps determines the cycle length of the cosine annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0, which corresponds
        to going from cos(0) to cos(pi)
        """
        super(CosineAnnealingLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class CosineAnnealingWarmRestartsLR(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_steps=0):
        super(CosineAnnealingWarmRestartsLR, self).__init__(optimizer, warmup_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min, last_epoch)

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersLinearLR(LRScheduler):
    """
    Scheduler that decays linearly.
    """

    def __init__(self, optimizer, training_steps, warmup_steps=0):
        """
        training_steps determines the cycle length of the linear annealing.

        It indicates the number of steps from 1.0 multiplier to 0.0
        """
        super(TransformersLinearLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._linear_lr)

    def _linear_lr(self, step):
        return max(0.0, float(self.training_steps - step) / float(max(1, self.training_steps)))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersCosineLR(LRScheduler):
    def __init__(self, optimizer, training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
                 warmup_steps: int = 0):
        super(TransformersCosineLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.num_cycles = num_cycles
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._cosine_lr, last_epoch)

    def _cosine_lr(self, step):
        progress = float(step) / float(max(1, self.training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersCosineWithHardRestartsLR(LRScheduler):
    def __init__(self, optimizer, training_steps: int, num_cycles: int = 1, last_epoch: int = -1,
                 warmup_steps: int = 0):
        super(TransformersCosineWithHardRestartsLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.num_cycles = num_cycles
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._cosine_with_hard_restarts_lr, last_epoch)

    def _cosine_with_hard_restarts_lr(self, step):
        progress = float(step) / float(max(1, self.training_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class TransformersPolynomialDecayLR(LRScheduler):
    def __init__(self, optimizer, training_steps, lr_end=1e-7, power=1.0, last_epoch=-1, warmup_steps=0):
        super(TransformersPolynomialDecayLR, self).__init__(optimizer, warmup_steps)
        self.training_steps = training_steps - warmup_steps
        self.lr_init = optimizer.defaults["lr"]
        self.lr_end = lr_end
        assert self.lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({self.lr_init})"
        self.power = power
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._polynomial_decay_lr, last_epoch)

    def _polynomial_decay_lr(self, step):
        if step > self.training_steps:
            return self.lr_end / self.lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = self.lr_init - self.lr_end
            decay_steps = self.training_steps
            pct_remaining = 1 - step / decay_steps
            decay = lr_range * pct_remaining ** self.power + self.lr_end
            return decay / self.lr_init  # as LambdaLR multiplies by lr_init

    def train_adjust(self):
        self.scheduler.step()

    def valid_adjust(self, metric):
        pass


class LinearIncreasingLR(LRScheduler):
    """
    线性提高学习率的调度器。
    学习率从 initial_lr 线性增加到 max_lr，在 total_epochs 内完成。
    如果定义了 warmup_steps，则在前 warmup_steps 内进行预热。
    """

    def __init__(self, optimizer, total_epochs, max_lr, warmup_steps=0):
        super(LinearIncreasingLR, self).__init__(optimizer, warmup_steps)
        self.total_epochs = total_epochs - warmup_steps  # 实际用于线性增加的epoch数
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.max_lrs = [max_lr] * len(self.initial_lrs)  # 假设所有参数组使用相同的max_lr
        self.current_epoch = 0

        if self.total_epochs <= 0:
            raise ValueError("total_epochs for LinearIncreasingLR must be > warmup_steps")
        for initial_lr, final_lr in zip(self.initial_lrs, self.max_lrs):
            if initial_lr >= final_lr:
                logger.warning(
                    f"Initial LR ({initial_lr}) is already >= max_lr ({final_lr}) "
                    f"for LinearIncreasingLR. LR will not increase."
                )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_lambda)

    def _lr_lambda(self, step):  # step 参数在这里代表当前的总训练步数，我们需要基于epoch调整
        # 这个 lambda 函数会在 warmup 之后被 LRScheduler 基类调用 (通过 self.scheduler.step())
        # 但更自然的 epoch 级别调整需要我们自己管理 current_epoch
        # 为了简单起见，我们让 train_adjust 或 valid_adjust 来驱动 epoch 级别的变化

        # 如果在 warmup 阶段，由基类的 warmup_scheduler 处理
        if self._is_lr_warming_up() or self.current_epoch == 0:  # 在 warmup 期间或第一个 epoch 保持初始（或 warmup 后）的学习率
            return 1.0  # warmup_scheduler 会乘以这个值

        if self.current_epoch > self.total_epochs:
            progress = 1.0
        else:
            progress = self.current_epoch / self.total_epochs

        # 计算每个参数组的学习率乘子
        # 注意：LambdaLR 的 lambda 函数返回的是乘子，而不是绝对学习率
        # 但由于我们在 init_optim 中每次都重新创建 optimizer 和 scheduler，
        # self.initial_lrs 会是当前阶段配置的 lr。
        # 这里的设计可以更精细，但为了演示，我们直接修改优化器中的学习率。
        # 一个更符合 LambdaLR 的做法是计算相对于 initial_lr 的比例。

        # 此处我们返回一个固定的1.0，实际的LR调整将在 train_adjust 中进行
        return 1.0

    def train_adjust(self):
        # 这个方法在每个训练步骤后被调用 (在 warmup 之后)
        # 对于 epoch 级别的调整，我们应该在 epoch 结束时调用。
        # BaseSystem 的 adjust_lr(metric) 是在 epoch 结束时被调用（用于验证指标）
        # 为了实现每个 epoch 增加，我们需要在 KGSFSystem 的训练循环中手动调用一个 epoch_step 方法，
        # 或者修改 BaseSystem 的逻辑。

        # 简化的做法：假设 train_adjust 被每个 epoch 调用一次（或在 KGSFSystem 中实现）
        # 这里我们先不修改 BaseSystem，而是假设这个调度器主要通过 valid_adjust（在每个epoch验证后）调整
        pass

    def valid_adjust(self, metric=None):
        # 在每个 epoch 的验证阶段之后调用
        super().valid_adjust(metric)  # 处理非 warmup 情况
        if self._is_lr_warming_up():
            return

        self.current_epoch += 1
        if self.current_epoch <= self.total_epochs:
            for i, (initial_lr, max_lr, param_group) in enumerate(
                    zip(self.initial_lrs, self.max_lrs, self.optimizer.param_groups)):
                if initial_lr >= max_lr:
                    new_lr = initial_lr  # 或者 max_lr，如果不想超过
                else:
                    progress = self.current_epoch / self.total_epochs
                    new_lr = initial_lr + progress * (max_lr - initial_lr)
                param_group['lr'] = new_lr
                logger.debug(f"Epoch {self.current_epoch}: Set LR for group {i} to {new_lr:.2e}")
        else:  # 已经达到或超过 total_epochs，学习率固定为 max_lr
            for i, (max_lr, param_group) in enumerate(zip(self.max_lrs, self.optimizer.param_groups)):
                param_group['lr'] = max_lr
                logger.debug(f"Epoch {self.current_epoch}: LR for group {i} maintained at max_lr {max_lr:.2e}")
