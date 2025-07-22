# ✅ 文件：moma_highway_env.py（已修复）
import numpy as np
from highway_env.envs.highway_env import HighwayEnv

class MOMAHighwayEnv(HighwayEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config)
        self.render_mode = render_mode

        # ✅ 确保 max_speed 存在
        if "max_speed" not in self.config:
            self.config["max_speed"] = self.config["reward_speed_range"][1]

        self.collision_penalty = -5.0
        self.risk_distance = 8.0
        self.risk_penalty = -0.5

    def _reward(self, action):
        ego = self.vehicle
        speed = np.clip(ego.speed, 0, self.config["max_speed"])
        speed_reward = speed / self.config["max_speed"]

        energy_reward = -0.05 * (np.abs(action) ** 2).mean()

        # --- ① 计算原始负惩罚总和（碰撞 + 距离） ---
        penalty = 0.0
        if ego.crashed:
            penalty += self.collision_penalty  # -5.0
        lead_vehicle = self._closest_vehicle_ahead(ego)
        if lead_vehicle is not None:
            dist = lead_vehicle.position[0] - ego.position[0]
            if dist < self.risk_distance:
                penalty += self.risk_penalty * (self.risk_distance - dist) / self.risk_distance  # -0.5 max

        # --- ② 负惩罚 → 正效用 [0,1] ---
        max_penalty = abs(self.collision_penalty) + abs(self.risk_penalty)  # 5.5
        safety_util = 1.0 + (penalty / max_penalty)  # penalty ∈ [-5.5, 0] → [0, 1]

        return np.array([speed_reward, energy_reward, safety_util])

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward_vec = self._reward(action)
        info.update({
            "speed_reward": reward_vec[0],
            "energy_reward": reward_vec[1],
            "safety_reward": reward_vec[2]
        })
        return obs, reward_vec, terminated, truncated, info

    def _closest_vehicle_ahead(self, ego):
        """返回当前车道上在 ego 前方最近的车辆，没有则返回 None"""
        lane = ego.lane_index
        vehicles = [
            v for v in self.road.vehicles
            if v is not ego and v.lane_index == lane and v.position[0] > ego.position[0]
        ]
        if not vehicles:
            return None
        return min(vehicles, key=lambda v: v.position[0] - ego.position[0])