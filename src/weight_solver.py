import numpy as np

class SensitivityGameSolver:
    """
    ✅ 三目标博弈权重求解器
    - 输入: f_vec = [f_speed, f_energy, f_safety]
    - 输出: w_vec = [w_speed, w_energy, w_safety]
    """

    def __init__(self, beta: float = 1.0, smooth_tau: float = 0.8):
        """
        :param beta: 控制目标敏感度放大系数 (beta>1 更强调高分目标)
        :param smooth_tau: 平滑系数 (0~1, 越大越平滑)
        """
        self.beta = beta
        self.smooth_tau = smooth_tau
        self.last_weights = None  # 记录上一次权重用于滑动平均

    def compute_weights(self, f_vec: np.ndarray) -> np.ndarray:
        """
        计算权重:
        w_i = f_i^beta / sum_j f_j^beta
        """
        # 避免负数 -> 取 max(f, 0)
        f_clipped = np.clip(f_vec, 1e-6, None)
        f_beta = f_clipped ** self.beta
        w_raw = f_beta / np.sum(f_beta)

        # 初始时直接返回
        if self.last_weights is None:
            self.last_weights = w_raw
            return w_raw

        # ✅ 平滑更新：避免权重抖动
        w_smooth = self.smooth_tau * w_raw + (1 - self.smooth_tau) * self.last_weights
        self.last_weights = w_smooth
        return w_smooth

    def reset(self):
        """重置历史权重"""
        self.last_weights = None


if __name__ == "__main__":
    # === 单元测试 ===
    solver = SensitivityGameSolver(beta=1.2, smooth_tau=0.7)

    # 模拟三目标得分 (速度奖励, 能耗奖励, 安全奖励)
    for t in range(10):
        f_speed = np.random.uniform(0, 1)
        f_energy = np.random.uniform(0, 1)
        f_safety = np.random.uniform(0, 1)
        f_vec = np.array([f_speed, f_energy, f_safety])

        w_vec = solver.compute_weights(f_vec)
        print(f"Step {t} | f_vec={f_vec.round(3)} -> w_vec={w_vec.round(3)}")
