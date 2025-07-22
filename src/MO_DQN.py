import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.DQN_Network import DQN_Network
from src.utils import ReplayBuffer


class MO_DQN:
    """
    ✅ Multi-Objective DQN
    - 网络包含 3 个分支（速度/能耗/安全）
    - ReplayBuffer 存储 reward_vec (3维)
    - 支持动态权重 w_vec (博弈求解)
    """

    def __init__(
        self,
        env,
        num_objectives=3,
        num_actions=5,
        observation_space_shape=(1, 1),
        replay_buffer_size=5000,
        batch_size=None,          # ✅ 允许直接给 batch_size
        batch_ratio=0.2,          # ✅ 兼容旧接口
        network_hidden_sizes=[128, 128],
        lr=1e-3,
        gamma=0.99,
        device=None,
    ):
        self.env = env
        self.num_objectives = num_objectives
        self.num_actions = num_actions

        # ✅ 兼容 obs_dim (Box不再 subscriptable)
        if hasattr(env.observation_space, "shape"):
            self.obs_dim = env.observation_space.shape[0]
        else:
            # 某些封装返回 tuple，需要解包
            self.obs_dim = env.observation_space[0].shape[0]

        self.gamma = gamma

        # ✅ 先保存 rb_size 再算 batch_size
        self.rb_size = replay_buffer_size
        self.batch_ratio = batch_ratio
        self.batch_size = batch_size if batch_size else int(self.rb_size * self.batch_ratio)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ 初始化 Q 网络
        self.q_network = DQN_Network(
            n_observations=self.obs_dim,
            n_actions=num_actions,
            n_objectives=num_objectives,
            hidden_sizes=network_hidden_sizes,
        ).to(self.device)

        self.target_q_network = DQN_Network(
            n_observations=self.obs_dim,
            n_actions=num_actions,
            n_objectives=num_objectives,
            hidden_sizes=network_hidden_sizes,
        ).to(self.device)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # ✅ 多目标 ReplayBuffer
        self.buffer = ReplayBuffer(
            buffer_size=replay_buffer_size,
            observation_space_shape=self.obs_dim,
            num_objectives=num_objectives,
            device=self.device,
            rng=torch.Generator().manual_seed(42),
        )

    def act(self, state, eps_greedy=True, epsilon=0.1):
        """
        ✅ epsilon-greedy 选动作
        """
        if eps_greedy and torch.rand(1) < epsilon:
            return torch.randint(0, self.num_actions, (1,)).item()
        else:
            q_values = self.q_network(state)      # shape = [1, num_objectives, num_actions]
            q_mean = q_values.mean(dim=1)         # 默认：用均值选动作（或传 w_vec）
            return torch.argmax(q_mean, dim=1).item()

    def __compute_loss(self, batch, w_vec=None):
        """
        ✅ 计算三目标 Q loss
        - batch: ReplayBuffer.sample()
        - w_vec: [3] 博弈权重
        """
        obs_batch = self.buffer.get_observations(batch)
        act_batch = self.buffer.get_actions(batch)
        next_obs_batch = self.buffer.get_next_obs(batch)
        reward_batch = self.buffer.get_rewards(batch)
        done_batch = self.buffer.get_termination_flag(batch)

        # ===== 当前 Q(s,a) =====
        q_pred_all = self.q_network(obs_batch)            # [B, 3, num_actions]
        q_pred = torch.gather(q_pred_all, dim=2, index=act_batch)  # [B, 3, 1]

        # ===== 下一个 Q'(s', a') =====
        with torch.no_grad():
            q_next_all = self.target_q_network(next_obs_batch)
            q_next_max = torch.max(q_next_all, dim=2, keepdim=True)[0]

        # ===== Bellman 目标 =====
        q_target = reward_batch.unsqueeze(-1) + self.gamma * q_next_max * (~done_batch).unsqueeze(-1)

        # ===== 三目标 MSE Loss =====
        per_obj_loss = F.mse_loss(q_pred, q_target, reduction="none")  # [B, 3, 1]

        # ✅ 支持博弈权重
        if w_vec is not None:
            w_tensor = torch.tensor(w_vec, device=self.device).reshape(1, -1, 1)
            loss = (per_obj_loss * w_tensor).sum()
        else:
            loss = per_obj_loss.mean()

        return loss

    def __update_weights(self, current_iteration, current_optimisation_iteration, inv_target_update_frequency):
        """
        ✅ Q 网络更新
        """
        batch = self.buffer.sample(self.batch_size)
        loss = self.__compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # ✅ 软更新目标网络
        if current_iteration % inv_target_update_frequency == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path="checkpoints/mo_dqn.pth"):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path="checkpoints/mo_dqn.pth"):
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    # ✅ 简单单元测试
    import mo_gymnasium as mo_gym
    env = mo_gym.make("moma-highway-env-v0")
    agent = MO_DQN(env=env)
    obs, _ = env.reset()
    obs = torch.tensor(obs.reshape(1, -1))   # ✅ 兼容 Box obs
    action = agent.act(obs)
    print("✅ 选动作:", action)
