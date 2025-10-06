import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import graphviz
import os

# biztosítjuk, hogy a Graphviz futtatható   #Nem akart lefutni sehogysem a Graphviz, és ezt talátam megoldásnak
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


#  Gyártási környezet definiálása


class FactoryEnv(gym.Env):
    def __init__(self): #Konstruktor 
        super().__init__()
        # Feladatok, gépek, operátorok - ezeket ismeri a modell úgymond egy szótár
        self.tasks = ["keveres", "szeval_1", "szeval_2", "szeval_3", "felmelegites"]
        self.machines = ["CNC1", "CNC2", "CNC3"]
        self.operators = ["Kati", "Bela"]

        self.action_space = gym.spaces.Discrete(len(self.tasks) * len(self.operators))   #Action space, tehát amit tehet az AI, minden lépésben egy számot választ a [0, n-1] tartományból. Pl:5 feladat × 2 operátor = 10
        self.observation_space = gym.spaces.Box(                                            #Observation space, vektorokkal adjuk meg, pl most 8 mert van 5 feladat 3 gépre
            low=0, high=1, shape=(len(self.tasks) + len(self.machines),), dtype=np.float32  #0 és 1 között vehet fel értéket 0 nincs kész 1 kész van
        )

        self.state = np.zeros(len(self.tasks) + len(self.machines))  #0ról indulunk tehát minden feladat nincs még kész és minden gép üres.
        self.done = False               #Self done annyit jelent hogy nincs kész még a feladat
        # operátorok foglaltsága
        self.operator_busy = {op: 0 for op in self.operators}    #Itt hoztam be hogy Béla tudjon egyszerre két gépen dolgozni késöbb a Step()-ben lesz fontos
 
    # Érvényes döntésvizsgálat - a feladat leírás alapján
    def valid_action(self, task, machine, operator):
        if task == "keveres" and machine != "CNC1":
            return False
        if task == "felmelegites" and machine != "CNC3":
            return False
        if task.startswith("szeval_") and machine not in ["CNC1", "CNC3", "CNC2"]:
            return False
        if operator == "Bela" and machine == "CNC1":
            return False

        # Kati egyszerre csak 1 gépet használhat
        if operator == "Kati" and self.operator_busy["Kati"] == 1:
            return False
        # Béla maximum 2 gépen dolgozhat egyszerre
        if operator == "Bela" and self.operator_busy["Bela"] >= 2:
            return False

        return True

    # RL lépés
    def step(self, action): 
        task_idx = action % len(self.tasks)  #Ugye az action az action térből, és segítségével visszanyerjük a taskot  (action=task*operátór)
        op_idx = action // len(self.tasks) #itt ugyanígy az operátórt
        task = self.tasks[task_idx]
        operator = self.operators[op_idx]

        # gépek hozzárendelése
        if task == "keveres" or task == "szeval_1":
            machine = "CNC1"
        elif task == "szeval_2" or task == "felmelegites":
            machine = "CNC3"
        else:
            machine = "CNC2"

        # akció értékelése
        if not self.valid_action(task, machine, operator):
            reward = -10
        else:
            reward = +5
            self.state[task_idx] = 1
            self.operator_busy[operator] += 1

        done = all(self.state[:len(self.tasks)])

        if done:
            self.operator_busy = {op: 0 for op in self.operators}

        return self.state, reward, done, False, {}

    # reset a környezet újraindításához ahoz szükséges hogy ujra tanuljunk
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(len(self.tasks) + len(self.machines))
        self.done = False
        self.operator_busy = {op: 0 for op in self.operators}
        return self.state, {}



#  Dinamikus folyamatábra generálása – tanult döntésekből https://graphviz.readthedocs.io/en/stable/manual.html

def draw_process_tree_parallel(actions_taken, env):   
    dot = graphviz.Digraph(comment="Tanult gyártási folyamat (párhuzamos)", format='png')
    dot.attr(rankdir="TB", size="10,8") #TB top to bottom, 10,8 a méretarány

    dot.node("Start", "Gyártás kezdete", shape="ellipse", style="filled", fillcolor="lightblue")  #Az end és a Start ugye fixek, azokat létrehoztuk ez nem változik soha
    dot.node("End", "Gyártás vége", shape="ellipse", style="filled", fillcolor="lightblue")

    # 1 lépés: szétválogatjuk a döntéseket idő szerint
    # feltételezzük, hogy a PPO minden step egy “időegység”
    # ha több operátor egyszerre dolgozik, azokat egy csoportba rakjuk
    time_groups = []
    current_group = []
    last_done = set()

    for i, action in enumerate(actions_taken):
        task_idx = action % len(env.tasks)
        op_idx = action // len(env.tasks)
        task = env.tasks[task_idx]
        operator = env.operators[op_idx]

        # gép meghatározása
        if task == "keveres" or task == "szeval_1":
            machine = "CNC1"
        elif task == "szeval_2" or task == "felmelegites":
            machine = "CNC3"
        else:
            machine = "CNC2"

        # ha ugyanabban az időben másik ember másik gépen dolgozik → párhuzamos
        if operator not in last_done:
            current_group.append((i, task, machine, operator))
            last_done.add(operator)
        else:
            time_groups.append(current_group)
            current_group = [(i, task, machine, operator)]
            last_done = {operator}

    if current_group:
        time_groups.append(current_group)

    #  2️ lépés: rajzolás 
    prev_nodes = ["Start"]

    for t, group in enumerate(time_groups):
        # ha több feladat fut egyszerre, azokat ugyanabba a rank csoportba rakjuk
        with dot.subgraph() as s:
            s.attr(rank='same')
            for i, task, machine, operator in group:
                node_name = f"step{i}"
                label = f"{task.capitalize()}\n({machine} - {operator})"
                s.node(node_name, label, shape="box", style="rounded,filled", fillcolor="lightgreen")
                # minden előző szintről idelinkelünk
                for prev in prev_nodes:
                    dot.edge(prev, node_name)
        prev_nodes = [f"step{i}" for i, *_ in group]

    # utolsó szint a végponthoz csatlakozik
    for n in prev_nodes:
        dot.edge(n, "End")

    dot.render("gyartasi_folyamatfa_AI_PARHUZAMOS", view=False)
    print(" Automatikusan felismert párhuzamos folyamatfa létrehozva: gyartasi_folyamatfa_AI_PARHUZAMOS.png")



# Fő program

if __name__ == "__main__":
    env = FactoryEnv()
    print("Környezet inicializálva ")
    obs = env.reset()
    print("Kezdő állapot:", obs)

    # manuális lépések megfigyeléshez
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        print(f"Lépés {i+1}: akció={action}, jutalom={reward}, kész={done}, foglaltság={env.operator_busy}")
        if done:
            print("Epizód vége ")
            break

    # PPO tanulás
    print("\n  PPO modell tanítása indul...")
    env = DummyVecEnv([lambda: FactoryEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000) 
    print("Tanítás kész")

    # Szimuláció a tanult modell alapján
    actions_taken = []
    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)
        actions_taken.append(int(action[0]))
        obs, reward, done, info = env.step(action)
        if done:
            break

    # Folyamatfa generálás
    draw_process_tree_parallel(actions_taken, FactoryEnv())
