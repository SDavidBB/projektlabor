import os
import csv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Graphviz PATH (ha nálad kell)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# -------------------------
#  Gyártási környezet
# -------------------------
class FactoryEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.tasks = ["keveres", "szeval_1", "szeval_2", "szeval_3", "felmelegites"]
        self.machines = ["CNC1", "CNC2", "CNC3"]
        self.operators = ["Kati", "Bela"]

        self.action_space = gym.spaces.Discrete(len(self.tasks) * len(self.operators))
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(len(self.tasks) + len(self.machines),),
            dtype=np.float32
        )

        # állapot
        self.state = np.zeros(len(self.tasks) + len(self.machines), dtype=np.float32)

        # előfeltételek (DAG)
        self.task_index = {t: i for i, t in enumerate(self.tasks)}
        self.prereq = {
            "szeval_2": ["szeval_1"],
            "szeval_3": ["szeval_2"],
        }

        # reward paraméterek
        self.sequence_bonus = 5

        # epizód-limit, hogy biztosan záródjon epizód (különben a Monitor üres maradhat)
        self.max_steps = 50
        self.steps = 0

    # --- segédfüggvények ---
    def prereq_satisfied(self, task: str) -> bool:
        reqs = self.prereq.get(task, [])
        return all(self.state[self.task_index[r]] == 1 for r in reqs)

    def next_required_split_task(self):
        for t in ["szeval_1", "szeval_2", "szeval_3"]:
            if self.state[self.task_index[t]] == 0:
                return t
        return None

    def infer_machine(self, task: str) -> str:
        if task in ("keveres", "szeval_1"):
            return "CNC1"
        elif task in ("szeval_2", "felmelegites"):
            return "CNC3"
        else:
            return "CNC2"

    def valid_action(self, task, machine, operator) -> bool:
        # gépszabályok
        if task == "keveres" and machine != "CNC1":
            return False
        if task == "felmelegites" and machine != "CNC3":
            return False
        if task.startswith("szeval_") and machine not in ["CNC1", "CNC2", "CNC3"]:
            return False
        if operator == "Bela" and machine == "CNC1":
            return False

        # előfeltételek
        if not self.prereq_satisfied(task):
            return False

        # FONTOS: „busy” korlátot itt nem érvényesítünk,
        # mert egy step = egy akció, nincs „egylépésen belüli” párhuzam.
        return True

    # --- RL API ---
    def step(self, action):
        self.steps += 1

        task_idx = action % len(self.tasks)
        op_idx = action // len(self.tasks)
        task = self.tasks[task_idx]
        operator = self.operators[op_idx]
        machine = self.infer_machine(task)

        # érvényesség
        if not self.valid_action(task, machine, operator):
            # enyhe büntetés az érvénytelenért
            reward = -1.0
            terminated = False
            truncated = self.steps >= self.max_steps
            return self.state, reward, terminated, truncated, {}

        # ha már kész volt
        if self.state[task_idx] == 1:
            reward = -20.0
            terminated = False
            truncated = self.steps >= self.max_steps
            return self.state, reward, terminated, truncated, {}

        # érvényes, új feladat
        reward = 1.0

        # sorrend-bónusz (szeval_1 -> szeval_2 -> szeval_3)
        next_req = self.next_required_split_task()
        if next_req is not None and task == next_req:
            reward += self.sequence_bonus

        # állapot frissítése
        self.state[task_idx] = 1.0

        # epizód vége logika
        terminated = bool(all(self.state[:len(self.tasks)]))
        truncated = self.steps >= self.max_steps

        # ha minden kész: nagy bónusz
        if terminated:
            reward += 100.0

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state[:] = 0.0
        self.steps = 0
        return self.state, {}


# -------------------------
#  Szöveges ütemező
# -------------------------
def pretty_task_name(task: str) -> str:
    mapping = {
        "keveres": "Keverés",
        "felmelegites": "Felmelegítés",
        "szeval_1": "Szétválasztás 1",
        "szeval_2": "Szétválasztás 2",
        "szeval_3": "Szétválasztás 3",
    }
    return mapping.get(task, task.capitalize())

def infer_machine(task: str) -> str:
    if task in ("keveres", "szeval_1"):
        return "CNC1"
    elif task in ("szeval_2", "felmelegites"):
        return "CNC3"
    else:
        return "CNC2"

def actions_to_timeslices(actions, env: FactoryEnv):
    shadow = FactoryEnv(); shadow.reset()
    slice_map = {op: None for op in env.operators}
    used_ops = set()
    timeslices = []

    for a in actions:
        task_idx = int(a) % len(env.tasks)
        op_idx   = int(a) // len(env.tasks)
        task     = env.tasks[task_idx]
        op       = env.operators[op_idx]
        machine  = infer_machine(task)

        # ha ugyanabban a szeletben már kapott az op, zárunk
        if op in used_ops:
            for v in slice_map.values():
                if v is None: 
                    continue
                t, _ = v
                shadow.state[shadow.task_index[t]] = 1
            timeslices.append(slice_map)
            slice_map = {o: None for o in env.operators}
            used_ops.clear()

        # kihagyjuk a már kész/érvénytelen lépéseket
        if shadow.state[shadow.task_index[task]] == 1:
            continue
        if not shadow.prereq_satisfied(task):
            continue

        slice_map[op] = (task, machine)
        used_ops.add(op)

        if len(used_ops) == len(env.operators):
            for v in slice_map.values():
                if v is None: 
                    continue
                t, _ = v
                shadow.state[shadow.task_index[t]] = 1
            timeslices.append(slice_map)
            slice_map = {o: None for o in env.operators}
            used_ops.clear()

    if any(v is not None for v in slice_map.values()):
        for v in slice_map.values():
            if v is None: 
                continue
            t, _ = v
            shadow.state[shadow.task_index[t]] = 1
        timeslices.append(slice_map)

    return timeslices

def print_text_schedule(actions, env: FactoryEnv):
    slices = actions_to_timeslices(actions, env)
    if not slices:
        print("Nincs ütemezhető művelet.")
        return

    col_names = env.operators
    width_left, width_col = 4, 28
    sep = "  |  "
    for i, sl in enumerate(slices, start=1):
        parts = []
        for op in col_names:
            if sl.get(op):
                task, machine = sl[op]
                txt = f"{op}: {pretty_task_name(task)} ({machine})"
            else:
                txt = f"{op}: –"
            parts.append(txt.ljust(width_col))
        line = f"{str(i)+'.':<{width_left}}" + sep.join(parts)
        print(line)


# -------------------------
#  Tanulási görbe rajzoló
# -------------------------
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


# -------------------------
#  Fő program
# -------------------------
if __name__ == "__main__":
    # Log könyvtár + monitorozott környezet
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    monitor_path = os.path.join(log_dir, "monitor.csv")

    def make_env():
        e = FactoryEnv()
        # a Monitor ide írja az epizódokat
        return Monitor(e, monitor_path)

    # (opcionális) kézi próba
    raw = FactoryEnv()
    obs, _ = raw.reset()
    print("Kézi próba – kezdő állapot:", obs)

    # PPO tanítás monitorozott környezetben
    print("\nPPO tanítás indul…")
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, learning_rate=1e-4, ent_coef=0.01, verbose=1)
    model.learn(total_timesteps=50_000)  # elég, hogy több epizód is lezáruljon
    print("Tanítás kész.")

    # Tanulási görbe
    plot_learning_curve(monitor_path, out_png="rewards.png")

    # Kiértékelés + szöveges ütemezés
    actions_taken = []
    obs = env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(int(action[0]))
        obs, reward, done, info = env.step(action)
        if bool(done[0]):  # VecEnv: bármelyik env lezárt
            break

    print("\nEgyszerű ütemezés (időszeletek):")
    print_text_schedule(actions_taken, FactoryEnv())
