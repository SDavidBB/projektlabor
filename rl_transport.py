from typing import Optional
import numpy as np
from random import randint, random
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import csv

class TransportEnvironment(gym.Env):
    max_transport_cost = 10.0
    min_transport_cost = 2.3
    
    remote_transport_cost = max_transport_cost * 100
    
    min_demand = 500
    max_demand = 1200

    min_supply = 200
    max_supply= 1000

    num_warehouses = 4
    num_destinations = 6

    def _init_data(self):
        self.warehouse_supplies = [randint(self.min_supply, self.max_supply)
                              for _ in range(self.num_warehouses)]
        self.warehouse_supplies[-1] = 2**62  # Remote warehouse with large supply
        
        self.destination_demands = [randint(self.min_demand, self.max_demand)
                               for _ in range(self.num_destinations)]
        self.cost_matrix = [[random() * (self.max_transport_cost - self.min_transport_cost) + self.min_transport_cost
                        for _ in range(self.num_destinations)] for _ in range(self.num_warehouses)]
        self.cost_matrix[-1] = [self.remote_transport_cost for _ in range(self.num_destinations)]  # Remote warehouse costs

    def __init__(self,):
        super(TransportEnvironment, self).__init__()
        
        self.action_space = gym.spaces.Box(low=0, high=2**62,
            shape=(self.num_warehouses, self.num_destinations),
            dtype=float)  # Example: 4 warehouses * 6 destinations
        self.observation_space = gym.spaces.Dict(
            {
                "supply": gym.spaces.Box(low=0, high=self.max_supply,
                    shape=(self.num_warehouses,), dtype=float),  # supply observation
                "demand": gym.spaces.Box(low=0, high=self.max_demand,   
                    shape=(self.num_destinations,), dtype=float),  # demand observation
                "cost": gym.spaces.Box(low=0, high=self.max_transport_cost,
                    shape=(self.num_warehouses, self.num_destinations), dtype=float),  # cost observation
            }
        ) 
    def reset(self, seed : Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_data()
        observation = {
            "supply": np.array(self.warehouse_supplies, dtype=float),
            "demand": np.array(self.destination_demands, dtype=float),
            "cost": np.array(self.cost_matrix, dtype=float),
        }
        return observation, {}  # return initial observation
    
    def step(self, action):
        cost = 0
        for i in range(self.num_warehouses):
            for j in range(self.num_destinations):
                cost += action[i][j] * self.cost_matrix[i][j]
            if action[i].sum() > self.warehouse_supplies[i]:
                cost += 1000 * (action[i].sum() - self.warehouse_supplies[i])  # Penalty for exceeding supply
        for j in range(self.num_destinations):
            if action[:, j].sum() < self.destination_demands[j]:
                cost += 1 * (self.destination_demands[j] - action[:, j].sum())  # Penalty for not meeting demand
        reward = -cost
        done = True
        info = {}
        return {}, reward, done, done, info

def plot_learning_curve(monitor_csv_path: str, out_png: str = "rewards.png"):
    ep_rewards, ep_idx = [], []

    if not os.path.exists(monitor_csv_path):
        print(f"Nem található monitor fájl: {monitor_csv_path}")
        return

    with open(monitor_csv_path, "r", newline="") as f:
        _ = f.readline()  # JSON header
        reader = csv.DictReader(f)
        i = 1
        for row in reader:
            try:
                r = float(row.get("r", "nan"))
            except ValueError:
                continue
            if np.isfinite(r):
                ep_rewards.append(r); ep_idx.append(i); i += 1

    if not ep_rewards:
        print("A monitor fájl üresnek tűnik (nem zárult epizód?).")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(ep_idx, ep_rewards)
    plt.xlabel("Epizód"); plt.ylabel("Összjutalom (return)")
    plt.title("PPO tanulási görbe (Monitor)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f" Tanulási görbe mentve: {out_png}")

if __name__ == "__main__":
    env = TransportEnvironment()
    obs, _ = env.reset()
    env = Monitor(env, filename="logs/transport_monitor.csv")
    print("Initial Observation:")
    print("Supply:", obs["supply"])
    print("Demand:", obs["demand"])
    print("Cost Matrix:\n", obs["cost"])
    model = PPO("MultiInputPolicy", env, verbose=1, device="cpu", learning_rate=0.003, ent_coef=0.01)
    model.learn(total_timesteps=1_000_00)
    model.save("models/transport_ppo")
    plot_learning_curve("logs/transport_monitor.csv", out_png="rewards.png")

def predict_transport(model_path: str, transport_data: gym.spaces.Dict):
    model = PPO.load(model_path)
    model.predict(transport_data)