import sys, os
import numpy as np
import pytest

# ✅ 自动添加 src 路径，保证能找到 weight_solver.py
CURRENT_DIR = os.path.dirname(__file__)  # tests/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)  # 优先加载 src/

from weight_solver import SensitivityGameSolver


@pytest.fixture
def solver():
    """每个测试都初始化一个solver，避免历史权重影响"""
    return SensitivityGameSolver(beta=1.2, smooth_tau=0.7)


def test_weights_sum_to_one(solver):
    f_vec = np.array([0.5, 0.3, 0.2])
    w = solver.compute_weights(f_vec)

    # ✅ 权重和应该=1
    assert np.isclose(np.sum(w), 1.0), f"权重总和错误: {np.sum(w)}"
    # ✅ 权重应该都是非负数
    assert np.all(w >= 0), "权重中有负数!"


def test_extreme_values_stability(solver):
    # 极端值 -> 应该不会崩溃
    f_vec = np.array([1e6, 1e-9, 0])
    w = solver.compute_weights(f_vec)
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= 0)


def test_smoothing_effect():
    """测试平滑机制是否生效"""
    solver = SensitivityGameSolver(beta=1.0, smooth_tau=0.5)

    f1 = np.array([1.0, 0.0, 0.0])
    w1 = solver.compute_weights(f1)

    f2 = np.array([0.0, 1.0, 0.0])
    w2 = solver.compute_weights(f2)

    # 如果完全替换，w2应该是 [0,1,0]，但平滑后不应完全变化
    assert 0 < w2[0] < 1, f"平滑无效，w2[0]={w2[0]}"
    assert 0 < w2[1] < 1, f"平滑无效，w2[1]={w2[1]}"


@pytest.mark.parametrize("f_vec,expected_idx", [
    (np.array([1.0, 0.2, 0.1]), 0),  # 速度最优
    (np.array([0.2, 1.0, 0.1]), 1),  # 能耗最优
    (np.array([0.2, 0.3, 1.0]), 2),  # 安全最优
])
def test_preference_correct(solver, f_vec, expected_idx):
    w = solver.compute_weights(f_vec)
    assert np.argmax(w) == expected_idx, f"预期{expected_idx}最大，实际{np.argmax(w)}"
