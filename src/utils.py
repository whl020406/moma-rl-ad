import torch
import numpy as np
from collections import namedtuple
import pandas as pd
from pymoo.indicators.hv import HV
from typing import List, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ReplayBuffer:
    """
    ✅ 多目标 ReplayBuffer
    - 存储: obs, action, next_obs, reward_vec(3), done_flag
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space_shape: int,
        num_objectives: int,
        device,
        rng: np.random.Generator,
        importance_sampling: bool = False,
        prioritise_crashes: bool = False,
    ):
        self.size = buffer_size
        self.num_objectives = num_objectives
        self.observation_space_size = observation_space_shape
        self.device = device
        self.rng = rng
        self.importance_sampling = importance_sampling
        self.prioritise_crashes = prioritise_crashes

        # +4: action, done_flag, importance_id(可选)
        self.buffer = torch.zeros(
            size=(self.size, self.observation_space_size * 2 + self.num_objectives + 3),
            device=self.device,
        )

        self.running_index = 0  # 当前写入位置
        self.num_elements = 0  # 当前 buffer 填充数量

    def push(self, obs, action, next_obs, reward_vec, done_flag, importance_id=None, num_samples: int = 1):
        """
        ✅ 录入一条数据：
        obs, action, next_obs, reward_vec(3目标), done_flag
        """
        assert num_samples >= 1
        if not self.importance_sampling:
            importance_id = torch.tensor([0], device=self.device)

        if num_samples == 1:
            elem = torch.cat(
                [
                    obs.flatten(),
                    torch.tensor([action], device=self.device),
                    next_obs.flatten(),
                    torch.as_tensor(reward_vec, device=self.device),
                    torch.tensor([done_flag], device=self.device),
                    importance_id,
                ]
            )
            self.buffer[self.running_index] = elem
            self.__increment_indices()
        else:
            for i in range(num_samples):
                elem = torch.cat(
                    [
                        obs[i].flatten(),
                        torch.tensor([action[i]], device=self.device),
                        next_obs[i].flatten(),
                        reward_vec[i],
                        torch.tensor([done_flag[i]], device=self.device),
                        torch.tensor([importance_id], device=self.device),
                    ]
                )
                self.buffer[self.running_index] = elem
                self.__increment_indices()

    def __increment_indices(self):
        self.running_index = (self.running_index + 1) % self.size
        if self.num_elements < self.size:
            self.num_elements += 1

    def sample(self, sample_size):
        sample_probs = (torch.ones(self.num_elements) / self.num_elements).to(self.device)
        if self.importance_sampling:
            sample_probs = self.compute_importance_sampling_probs()

        if self.prioritise_crashes:
            crashed_flag = self.buffer[: self.num_elements, -2].to(dtype=torch.bool)
            sample_probs[crashed_flag] = sample_probs[crashed_flag] * 2

        sample_probs = sample_probs / torch.sum(sample_probs)
        sample_probs = sample_probs.cpu().numpy()
        sample_indices = self.rng.choice(
            self.num_elements, p=sample_probs, size=max(1, round(sample_size)), replace=True
        )
        return self.buffer[sample_indices]

    def compute_importance_sampling_probs(self):
        imp_ids = self.buffer[: self.num_elements, -1]
        min_id = torch.min(imp_ids)
        probs = (imp_ids - min_id + 1)
        return probs

    # ========= 数据解析接口 =========
    def get_observations(self, samples):
        return samples[:, : self.observation_space_size]

    def get_actions(self, samples: torch.Tensor):
        return samples[:, self.observation_space_size].long().reshape(-1, 1, 1)

    def get_next_obs(self, samples):
        start = self.observation_space_size + 1
        end = self.observation_space_size * 2 + 1
        return samples[:, start:end]

    def get_rewards(self, samples):
        return samples[:, self.observation_space_size * 2 + 1 : -2]  # ✅ 三目标 reward_vec

    def get_termination_flag(self, samples):
        return samples[:, -2].flatten().bool()

    def get_importance_sampling_id(self, samples):
        return samples[:, -1].flatten()


class DataLogger:
    """
    ✅ 简单日志器 - 记录权重 & 其他统计数据
    """

    def __init__(self, logger_name: str, field_names: List[str]):
        self.tupleType = namedtuple(logger_name, field_names)
        self.tuple_list = []

    def add(self, *args, **kwargs):
        if isinstance(args, tuple) and len(args) == 1 and len(kwargs.values()) == 0:
            self.tuple_list.append(self.tupleType(*args[0]))
        elif isinstance(args, tuple) and len(args) == 0 and len(kwargs.values()) == 1:
            self.tuple_list.append(self.tupleType(*kwargs.values()))
        else:
            self.tuple_list.append(self.tupleType(*args, **kwargs))

    def to_dataframe(self):
        return pd.DataFrame(self.tuple_list)

    def save_csv(self, path="logs/weights_log.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)


def random_objective_weights(num_objectives: int, rng: np.random.Generator, device):
    random_weights = rng.random(num_objectives)
    random_weights = torch.tensor(random_weights / np.sum(random_weights), device=device)
    return random_weights


def calc_hypervolume(reference_point: np.ndarray = np.array([0, 0, 0]), reward_vector: np.ndarray = None):
    """
    ✅ 超体积 (Hypervolume) 指标
    - reference_point: 最差解
    - reward_vector: 多目标 reward 矩阵
    """
    assert reward_vector is not None, "必须提供 reward_vector!"
    reward_vector = reward_vector * (-1)  # 转为最小化问题
    ind = HV(ref_point=reference_point)
    return ind(reward_vector)
