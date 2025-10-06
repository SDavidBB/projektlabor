from stable_baselines3 import PPO   #Algoritmusok RL traininghez - PRO algoritmusok
from stable_baselines3.common.vec_env import DummyVecEnv  #Pro algoritmusok csak Vektorizált értékeket tudnak befogadni és ahoz hogy átalakítsuk vektorizáltá a környezett ez kell
import gymnasium as gym
import numpy as np

class BusinessProcessEnv(gym.Env):           #RL környezet
    def __init__(self):
        super(BusinessProcessEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 3 feladat
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(3,), dtype=np.int32) #3= vektorok száma(Feladatok) 0-10 ig (csak 1 et és 0-at használjuk 1=nincs kész 0=kész)
        self.state = np.array([1, 1, 1], dtype=np.int32)   #minden feladat elérhető 1= nincs kész(elérhető)
        self.done = False

    def step(self, action):
        reward = -1    #A lépséekért -1 reward jár (Múlik az idő)
        self.state[action] = 0   #pl [1,1,1] => [1,0,1]
        if np.sum(self.state) == 0:   #Ha [0,0,0] akkor kész vagyunk és 10 reward
            self.done = True
            reward = 10
        return self.state, reward, self.done, False, {}  # gymnasium 5 értéket ad vissza, state:pl [1,0,1], reward pl -1 vagy 9

    def reset(self, seed=None, options=None):   #Visszaálit minden [1,1,1]-re
        super().reset(seed=seed)
        self.state = np.array([1, 1, 1], dtype=np.int32)
        self.done = False
        return self.state, {}

env = DummyVecEnv([lambda: BusinessProcessEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("kész")