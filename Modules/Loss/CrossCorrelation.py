import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt


class LocalCrossCorrelation2D(nn.Module):
    def __init__(self, win=[9, 9]):
        """Initialize the Local Cross Correlation (LCC) model for 2D images

        Args:
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(LocalCrossCorrelation2D, self).__init__()
        self.win = win

    def set(self, win):
        self.win = win


    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]],
                                device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        return -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1


class WeightedLocalCrossCorrelation2D(nn.Module):
    def __init__(self, alpha=0.02, win=[9, 9]):
        """Initialize the WeightedL Local Cross Correlation (WLCC) model for 2D images

        Args:
            alpha (float, optional): The factor of the WLCC. Defaults to 0.02.
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(WeightedLocalCrossCorrelation2D, self).__init__()
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def set(self, alpha, win):
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through WLCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]],
                                device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        # calculating weight according the intensity difference
        P = self.normal.log_prob(torch.abs(I - J)).exp()
        weight = P / self.normal.log_prob(0).exp()

        dccp = weight + cc * (1 - weight)

        return -1.0 * torch.mean(dccp, dim=[1, 2, 3]) + 1

class NCC_vxm2D(nn.Module):
    """
    Local normalized cross correlation (2D), VXM 逻辑：
      - cc = (cross^2) / (I_var * J_var + 1e-5)
      - 返回标量 (对 B,C,H,W 全部平均)
    结构与 LocalCrossCorrelation2D 相同：__init__ / set / forward
    输入形状: [B, C, H, W]（支持多通道，逐通道分组卷积）
    """
    def __init__(self, win=[9, 9]):
        super().__init__()
        assert isinstance(win, (list, tuple)) and len(win) == 2, "win must be [kh, kw]"
        assert win[0] % 2 == 1 and win[1] % 2 == 1, "win elements should be odd"
        self.win = list(win)

    def set(self, win):
        assert isinstance(win, (list, tuple)) and len(win) == 2, "win must be [kh, kw]"
        assert win[0] % 2 == 1 and win[1] % 2 == 1, "win elements should be odd"
        self.win = list(win)

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        # I, J: [B, C, H, W]
        assert I.shape == J.shape, "I and J must have the same shape"
        B, C, H, W = I.shape
        kh, kw = self.win
        pad = (kw // 2, kh // 2)  # (padW, padH)

        # 分组卷积核: [C, 1, kh, kw]，groups=C 实现逐通道滑窗求和
        sum_filter = torch.ones(C, 1, kh, kw, device=I.device, dtype=I.dtype)

        # 基本量
        I2, J2, IJ = I * I, J * J, I * J
        groups = C

        I_sum  = F.conv2d(I,  sum_filter, padding=pad, groups=groups)
        J_sum  = F.conv2d(J,  sum_filter, padding=pad, groups=groups)
        I2_sum = F.conv2d(I2, sum_filter, padding=pad, groups=groups)
        J2_sum = F.conv2d(J2, sum_filter, padding=pad, groups=groups)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=pad, groups=groups)

        win_size = torch.tensor(kh * kw, device=I.device, dtype=I.dtype)

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = (cross * cross) / (I_var * J_var + 1e-5)

        # 与 VXM 保持一致：对全部元素取均值，返回一个标量
        return -cc.mean()
