import numpy as np
import torch
from src.MO_DQN import MO_DQN
from src.weight_solver import SensitivityGameSolver
from src.utils import DataLogger


class MOMA_DQN:
    """
    ✅ MOMA-DQN (Multi-Objective Multi-Agent DQN)
    - 调度多目标环境、博弈权重求解器、MO_DQN网络
    - 默认支持三目标 (速度, 能耗, 安全)
    """

    def __init__(
        self,
        env,
        num_objectives: int = 3,
        num_actions: int = 5,
        observation_space_shape=(1, 1),
        gamma: float = 0.99,
        replay_buffer_size: int = 5000,
        batch_size: int = None,  # ✅ 可选，默认用 ratio 算
        batch_ratio: float = 0.2,
        network_hidden_sizes=[128, 128],
        device=None,
        seed: int = 42,
        beta: float = 1.2,
        smooth_tau: float = 0.8,
        objective_names: list = ["speed", "energy", "safety"],
        reward_structure: str = "ego_reward",
        use_multi_dqn: bool = False,
        **kwargs
    ):
        """
        :param reward_structure: 奖励结构，可能的值有 "ego_reward", "mean_reward"
        """
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_objectives = num_objectives
        self.num_actions = num_actions
        self.gamma = gamma
        self.rb_size = replay_buffer_size          # ✅ 先定义 rb_size
        self.batch_ratio = batch_ratio
        self.batch_size = batch_size or int(self.rb_size * self.batch_ratio)

        self.observation_space_shape = observation_space_shape
        self.objective_names = objective_names
        self.reward_structure = reward_structure
        self.use_multi_dqn = use_multi_dqn

        # ✅ 初始化博弈权重求解器
        self.weight_solver = SensitivityGameSolver(beta=beta, smooth_tau=smooth_tau)

        # ✅ 初始化 DQN 策略 (MO_DQN 默认多目标)
        self.policy = MO_DQN(
            env=env,
            num_objectives=num_objectives,
            num_actions=num_actions,
            observation_space_shape=observation_space_shape,
            replay_buffer_size=replay_buffer_size,
            batch_size=self.batch_size,
            network_hidden_sizes=network_hidden_sizes,
            lr=1e-3,
            gamma=gamma,
            device=self.device,
        )

        # ✅ 记录权重变化日志 (方便画图)
        if self.num_objectives == 3:
            self.weight_logger = DataLogger("weight_logger", ["step", "w_speed", "w_energy", "w_safety"])
        elif self.num_objectives == 2:
            self.weight_logger = DataLogger("weight_logger", ["step", "w_obj1", "w_obj2"])
        else:
            raise ValueError("目前只支持 2 或 3 目标")

    def train(
        self,
        max_episodes: int = 100,
        max_steps_per_episode: int = 500,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        log_interval=10,
        **kwargs
    ):
        """
        ✅ 训练主循环
        - 每 step:
            1. 环境返回 reward_vec
            2. 博弈权重求解器更新 w_vec
            3. w_vec -> DQN 策略更新
        """
        epsilon = epsilon_start
        global_step = 0

        for ep in range(max_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state[0].reshape(1, -1), device=self.device)

            ep_reward = 0
            for t in range(max_steps_per_episode):
                # ✅ epsilon-greedy 选择动作
                action = self.policy.act(state, eps_greedy=True, epsilon=epsilon)

                # ✅ 环境交互
                next_state, reward_vec, terminated, truncated, info = self.env.step(action)

                # ✅ 转 np → 计算博弈权重
                reward_vec = np.array(reward_vec)
                w_vec = self.weight_solver.compute_weights(reward_vec)

                # ✅ 记录权重
                if self.num_objectives == 3:
                    self.weight_logger.add(global_step, w_vec[0], w_vec[1], w_vec[2])
                elif self.num_objectives == 2:
                    self.weight_logger.add(global_step, w_vec[0], w_vec[1])

                # ✅ 转 torch
                next_state = torch.tensor(next_state[0].reshape(1, -1), device=self.device)
                reward_tensor = torch.tensor(reward_vec, device=self.device)

                # ✅ 推入 ReplayBuffer ✅修复 push 参数顺序
                self.policy.buffer.push(
                    state,
                    torch.tensor([action], device=self.device),
                    next_state,
                    reward_tensor,
                    torch.tensor([terminated], device=self.device)
                )

                # ✅ DQN 更新
                if self.policy.buffer.num_elements >= self.policy.rb_size:
                    self.policy._MO_DQN__update_weights(
                        current_iteration=global_step,
                        current_optimisation_iteration=global_step,
                        inv_target_update_frequency=20,
                    )

                state = next_state
                ep_reward += np.dot(w_vec, reward_vec)  # 动态加权 reward

                global_step += 1
                if terminated or truncated:
                    break

            # ✅ epsilon 衰减
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if ep % log_interval == 0:
                print(
                    f"[Episode {ep}/{max_episodes}] Total Reward={ep_reward:.3f} | Eps={epsilon:.3f}"
                )

        print("✅ 训练完成")

    def save_weight_log(self, path="logs/weights.csv"):
        df = self.weight_logger.to_dataframe()
        df.to_csv(path, index=False)
        print(f"✅ 权重日志已保存: {path}")

    def evaluate(
        self,
        num_episodes: int = 20,
        hv_reference_point=None,
        episode_recording_interval: int = 1,
        video_name_prefix: str = "",
        video_location: str = "./",
        **kwargs,
    ):
        """
        ✅ 评估策略：
        - 不探索 (epsilon=0)
        - 记录 total_reward、三目标 reward、HV 指标
        - 返回 summary_logger (每episode) & detail_logger (每step)
        """

        from src.utils import DataLogger, calc_hypervolume

        # 如果没有给 reference_point，就默认 [0,0,0]
        if hv_reference_point is None:
            hv_reference_point = np.array([0.0, 0.0, 0.0])

        # 每episode记录（summary），每step记录（detail）
        summary_logger = DataLogger(
            "summary_logger",
            ["episode", "total_reward", "reward_speed", "reward_energy", "reward_safety", "hypervolume"],
        )
        detail_logger = DataLogger(
            "detail_logger",
            ["episode", "step", "reward_speed", "reward_energy", "reward_safety", "action"],
        )

        self.q_network.eval()  # ✅ 评估模式

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state[0].reshape(1, -1), device=self.device)
            ep_reward_vecs = []  # 每步的3维reward
            total_reward = 0.0

            for step in range(1000):  # 给定最大评估步数
                # ✅ 纯greedy
                action = self.policy.act(state, eps_greedy=False, epsilon=0.0)

                next_state, reward_vec, terminated, truncated, _ = self.env.step(action)
                next_state = torch.tensor(next_state[0].reshape(1, -1), device=self.device)
                reward_vec = np.array(reward_vec)

                # 累计
                total_reward += reward_vec.sum()
                ep_reward_vecs.append(reward_vec)

                # detail日志
                detail_logger.add(
                    ep,
                    step,
                    reward_vec[0],
                    reward_vec[1],
                    reward_vec[2],
                    action,
                )

                state = next_state
                if terminated or truncated:
                    break

            # ✅ episode 结束后 计算HV
            ep_reward_vecs = np.array(ep_reward_vecs)
            if ep_reward_vecs.shape[0] > 0:
                hv = calc_hypervolume(reference_point=hv_reference_point, reward_vector=ep_reward_vecs)
                r_speed = ep_reward_vecs[:, 0].sum()
                r_energy = ep_reward_vecs[:, 1].sum()
                r_safety = ep_reward_vecs[:, 2].sum()
            else:
                hv = 0.0
                r_speed, r_energy, r_safety = 0.0, 0.0, 0.0

            # ✅ summary日志
            summary_logger.add(
                ep,
                total_reward,
                r_speed,
                r_energy,
                r_safety,
                hv,
            )

        self.q_network.train()  # ✅ 恢复训练模式

        return summary_logger.to_dataframe(), detail_logger.to_dataframe()


if __name__ == "__main__":
    # ====== 简单单元测试 ======
    import mo_gymnasium as mo_gym
    env = mo_gym.make("moma-highway-env-v0", render_mode=None)

    agent = MOMA_DQN(env)
    agent.train(max_episodes=5, max_steps_per_episode=100)
    agent.save_weight_log("weight_log.csv")
