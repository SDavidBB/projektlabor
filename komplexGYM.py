import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import graphviz
import os

# biztosítjuk, hogy a Graphviz futtatható   #Nem akart lefutni sehogysem a Graphviz, és ezt talátam megoldásnak
# ÚJ RÉSZ: javított PATH (ne legyen szóköz a "C:\" és "Program" között) – csak a biztonság kedvéért
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


#  Gyártási környezet definiálása
class FactoryEnv(gym.Env):
    def __init__(self): #Konstruktor 
        super().__init__()
        # Feladatok, gépek, operátorok - ezeket ismeri a modell úgymond egy szótár
        self.tasks = ["keveres", "szeval_1", "szeval_2", "szeval_3", "felmelegites"]
        self.machines = ["CNC1", "CNC2", "CNC3"]
        self.operators = ["Kati", "Bela"]

        self.taskorder = []  #Ez a lista fogja tárolni a feladatok sorrendjét, hogy majd később költséget számoljunk
        # selejt arány definiálása
        for task in self.tasks:
            if task.startswith("felmelegites"):
                setattr(self, f"{task}_defect_rate", 0.1)  # 10% selejt a felmelegítésnél
            else:
                setattr(self, f"{task}_defect_rate", 0.0)  # egyébként nincs selejt
                
        self.action_space = gym.spaces.Discrete(len(self.tasks) * len(self.operators))   #Action space, tehát amit tehet az AI, minden lépésben egy számot választ a [0, n-1] tartományból. Pl:5 feladat × 2 operátor = 10
        self.observation_space = gym.spaces.Box(                                            #Observation space, vektorokkal adjuk meg, pl most 8 mert van 5 feladat 3 gépre
            low=0, high=1, shape=(len(self.tasks) + len(self.machines),), dtype=np.float32  #0 és 1 között vehet fel értéket 0 nincs kész 1 kész van
        )

        self.state = np.zeros(len(self.tasks) + len(self.machines))  #0ról indulunk tehát minden feladat nincs még kész és minden gép üres.
        self.done = False               #Self done annyit jelent hogy nincs kész még a feladat
        # operátorok foglaltsága
        self.operator_busy = {op: 0 for op in self.operators}    #Itt hoztam be hogy Béla tudjon egyszerre két gépen dolgozni késöbb a Step()-ben lesz fontos

        # ÚJ RÉSZ: feladat-index térkép és előfeltételek (DAG)
        # ez kényszeríti a sorrendet: szeval_1 → szeval_2 → szeval_3.
        self.task_index = {t: i for i, t in enumerate(self.tasks)}
        self.prereq = {
            "szeval_2": ["szeval_1"],
            "szeval_3": ["szeval_2"],
            # "felmelegites": ["szeval_3"],
        }

        # ÚJ RÉSZ: sorrend-bónusz paraméter
        # ha pont a következő kötelező szétválasztási lépést választja, +5 jutalom
        self.sequence_bonus = 5
 
    # ÚJ RÉSZ: előfeltételek ellenőrzése
    # csak akkor engedünk egy taskot, ha minden előfeltétele már kész.
    def prereq_satisfied(self, task: str) -> bool:
        reqs = self.prereq.get(task, [])
        return all(self.state[self.task_index[r]] == 1 for r in reqs)

    # ÚJ RÉSZ: mi a „következő elvárt” szétválasztási lépés a jelen állapotban
    # ezt használjuk a sequence bonus kiszámításához
    def next_required_split_task(self):
        for t in ["szeval_1", "szeval_2", "szeval_3"]:
            if self.state[self.task_index[t]] == 0:
                return t
        return None  # mind kész

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

        # ÚJ RÉSZ: előfeltétel-szabályok érvényesítése
        # tiltjuk a rossz sorrendet (pl. szeval_3 nem mehet szeval_2 előtt).
        if not self.prereq_satisfied(task):
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
        self.taskorder.append(task)  # várható költség számoláshoz eltároljuk a sorrendet
        # gépek hozzárendelése
        if task == "keveres" or task == "szeval_1":
            machine = "CNC1"
        elif task == "szeval_2" or task == "felmelegites":
            machine = "CNC3"
        else:
            machine = "CNC2"

        # akció értékelése
        if not self.valid_action(task, machine, operator):
            reward = -1
            # 🔹 ÚJ RÉSZ: azonnali visszatérés invalid akciónál (nem módosítunk state-en)
            # Magyarázat: így a PPO nem tud "félre tanulni" tiltott lépéseket.
            return self.state, reward, False, False, {}

        # ÚJ RÉSZ: ha a feladat már kész, azonnal elutasítjuk (early return)
        # nincs állapotváltozás, erős büntetés; ettől tűnnek el a duplikációk a fáról.
        if self.state[task_idx] == 1:
            reward = -20   # már elvégzett feladatért negatív jutalom
            return self.state, reward, False, False, {}

        # új, érvényes feladat
        reward = +1   # új, érvényes feladatért pozitív jutalom

        # ÚJ RÉSZ: sorrend-bónusz a szétválasztás helyes következő lépéséért
        # így preferálja a szeval_1 → szeval_2 → szeval_3 sorrendet
        next_req = self.next_required_split_task()
        if next_req is not None and task == next_req:
            reward += self.sequence_bonus

        self.state[task_idx] = 1
        self.operator_busy[operator] += 1

        done = all(self.state[:len(self.tasks)])

        # ha minden feladat kész → extra jutalom
        if done:
            reward += 1000
            reward -= self.cost_estimate()  # levonjuk a költséget a végső jutalomból
            self.operator_busy = {op: 0 for op in self.operators}

        return self.state, reward, done, False, {}

    # reset a környezet újraindításához ahoz szükséges hogy ujra tanuljunk
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(len(self.tasks) + len(self.machines))
        self.done = False
        self.operator_busy = {op: 0 for op in self.operators}
        return self.state, {}
    
    def cost_estimate(self)-> float:
        # egyszerű költségmodell: minden feladat 10 egység, minden selejt 5 egység
        base_cost = 10
        total_cost = 0.0
        for task in self.taskorder:
            total_cost += base_cost
            defect_rate = getattr(self, f"{task}_defect_rate", 0.0)
            defect_mult = 1.0 / (1.0 - defect_rate) if defect_rate > 0 else 1.0
            total_cost *= defect_mult
        return total_cost


# ÚJ RÉSZ: Egyszerű szöveges ütemezés (időszeletek soronként)
#  A PPO egy akciót ad lépésenként. Itt "időszeletekbe" csoportosítjuk az akciókat:
# egy szeletben minden operátor legfeljebb 1 feladatot kaphat → párhuzamos végrehajtás egyszerűen.

# ÚJ RÉSZ: feladatnév "szépítése" magyar ékezetekkel (csak a kiíráshoz)
def pretty_task_name(task: str) -> str:
    mapping = {
        "keveres": "Keverés",
        "felmelegites": "Felmelegítés",
        "szeval_1": "Szétválasztás 1",
        "szeval_2": "Szétválasztás 2",
        "szeval_3": "Szétválasztás 3",
    }
    return mapping.get(task, task.capitalize())

# ÚJ RÉSZ: gép hozzárendelés (ugyanaz a logika mint a step-ben) – külön használjuk a kiíráshoz
def infer_machine(task: str) -> str:
    if task in ("keveres", "szeval_1"):
        return "CNC1"
    elif task in ("szeval_2", "felmelegites"):
        return "CNC3"
    else:
        return "CNC2"

# ÚJ RÉSZ: validáció a szöveges ütemezéshez – NEM vesszük figyelembe a "busy" szabályt,
# csak a gép–feladathoz illeszkedést és az előfeltételeket. Így Kati minden új szeletben tud dolgozni.
def valid_for_schedule(shadow: FactoryEnv, task: str, machine: str, operator: str) -> bool:
    if task == "keveres" and machine != "CNC1":
        return False
    if task == "felmelegites" and machine != "CNC3":
        return False
    if task.startswith("szeval_") and machine not in ["CNC1", "CNC3", "CNC2"]:
        return False
    # előfeltételek
    return shadow.prereq_satisfied(task)

# ÚJ RÉSZ: akciók időszeletek (egy szelet = egyszerre végrehajtott feladatok, op-onként max 1)
def actions_to_timeslices(actions, env: FactoryEnv):
    shadow = FactoryEnv(); shadow.reset()
    # aktuális szelet: op → (task, machine)
    slice_map = {op: None for op in env.operators}
    used_ops = set()
    timeslices = []  # lista: dict(op -> (task, machine) VAGY None)

    for a in actions:
        task_idx = int(a) % len(env.tasks)
        op_idx   = int(a) // len(env.tasks)
        task     = env.tasks[task_idx]
        op       = env.operators[op_idx]
        machine  = infer_machine(task)

        # ha az adott op már kapott a mostani szeletben, lezárjuk a szeletet és újat kezdünk
        if op in used_ops:
            # szelet lezárása → állapot frissítése
            for v in slice_map.values():
                if v is None: 
                    continue
                t, m = v
                shadow.state[shadow.task_index[t]] = 1
            timeslices.append(slice_map)
            slice_map = {o: None for o in env.operators}
            used_ops = set()

        # hagyjuk ki azokat a lépéseket, amelyek már teljesített feladatot céloznak vagy érvénytelenek
        if shadow.state[shadow.task_index[task]] == 1:
            continue
        if not valid_for_schedule(shadow, task, machine, op):
            continue

        slice_map[op] = (task, machine)
        used_ops.add(op)

        # ha minden operátor kapott feladatot, zárjuk a szeletet
        if len(used_ops) == len(env.operators):
            for v in slice_map.values():
                if v is None:
                    continue
                t, m = v
                shadow.state[shadow.task_index[t]] = 1
            timeslices.append(slice_map)
            slice_map = {o: None for o in env.operators}
            used_ops = set()

    # maradék szelet lezárása (ha van benne bármi)
    if any(v is not None for v in slice_map.values()):
        for v in slice_map.values():
            if v is None:
                continue
            t, m = v
            shadow.state[shadow.task_index[t]] = 1
        timeslices.append(slice_map)

    return timeslices

# ÚJ RÉSZ: szöveges kiírás soronként
def print_text_schedule(actions, env: FactoryEnv):
    slices = actions_to_timeslices(actions, env)
    if not slices:
        print("Nincs ütemezhető művelet.")
        return

    # oszlop-szélességekhez egy kis formázás
    col_names = env.operators
    width_left  = 4
    width_col   = 28  # egy oszlop szélessége (név + feladat)
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
    model = PPO("MlpPolicy", env, learning_rate=0.0001, ent_coef=0.01, verbose=1)
    model.learn(total_timesteps=200000) 
    print("Tanítás kész")

    # Szimuláció a tanult modell alapján
    actions_taken = []
    obs = env.reset()
    for _ in range(50):  # ÚJ RÉSZ: több lépés, hogy „végigérjen”
        # ÚJ RÉSZ: kis sztochasztika, hogy ne ragadjon be egy akcióba
        action, _ = model.predict(obs, deterministic=False)
        actions_taken.append(int(action[0]))
        obs, reward, done, info = env.step(action)
        if done:
            break

    # ÚJ RÉSZ: Egyszerű szöveges ütemezés kiírása (ez helyettesíti a vizuális ábrát)
    print("\nEgyszerű ütemezés (időszeletek):")
    print_text_schedule(actions_taken, FactoryEnv())
