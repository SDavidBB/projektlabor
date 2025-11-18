import random
from typing import List, Dict, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None

import pulp as pl


def random_demands(total: int = 4000, n: int = 6, min_each: int = 200) -> List[int]:

    base = [min_each] * n
    rem = total - min_each * n
    if rem < 0:
        raise ValueError("A total kisebb, mint n*min_each.")
    # n-1 véletlen vágópont 0..rem között -> nemnegatív egész kompozíció
    points = sorted([0] + [random.randint(0, rem) for _ in range(n - 1)] + [rem])
    extra = [points[i + 1] - points[i] for i in range(n)]
    return [base[i] + extra[i] for i in range(n)]


def generate_costs(warehouses: List[str], destinations: List[str]) -> Dict[str, Dict[str, int]]:
    costs = {w: {} for w in warehouses}
    # A, B, C költségei
    for w in ["A", "B", "C"]:
        for d in destinations:
            costs[w][d] = random.randint(10, 50)
    # D költsége legyen minden célban szigorúan a legnagyobb
    for d in destinations:
        m = max(costs["A"][d], costs["B"][d], costs["C"][d])
        costs["D"][d] = m + random.randint(1, 25)
    return costs


def print_table(matrix, row_labels, col_labels, title=None, decimals=2):
    if title:
        print(title)
    if pd is not None:
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        # Összegek
        df["Összesen"] = df.sum(axis=1)
        total_row = df.sum(axis=0)
        df = df.round(decimals)
        print(df.to_string())
        print()
    else:
        
        widths = [max(len(c), 10) for c in col_labels]
        roww = max(max(len(r) for r in row_labels), 8)
        header = " " * (roww + 2) + " ".join(c.rjust(w) for c, w in zip(col_labels, widths)) + "   Összesen"
        print(header)
        for r_label, row in zip(row_labels, matrix):
            osz = sum(row)
            cells = " ".join(f"{v:.{decimals}f}".rjust(w) for v, w in zip(row, widths))
            print(r_label.rjust(roww), " ", cells, f"   {osz:.{decimals}f}")
        col_sums = [sum(matrix[i][j] for i in range(len(row_labels))) for j in range(len(col_labels))]
        line = "—" * (roww + 2 + sum(widths) + 10 + (len(widths) - 1))
        print(line)
        print("Összesen".rjust(roww), " ",
              " ".join(f"{s:.{decimals}f}".rjust(w) for s, w in zip(col_sums, widths)),
              f"   {sum(col_sums):.{decimals}f}")
        print()


def main(seed: int = None):
    #Opcionális véletlenmag a reprodukálhatósághoz
    if seed is not None:
        random.seed(seed)

    warehouses = ["A", "B", "C", "D"]
    destinations = ["1", "2", "3", "4", "5", "6"]

    #Költségek (D mindenhol legdrágább)
    costs = generate_costs(warehouses, destinations)

    #Készletek
    supplies: Dict[str, int] = {
        "A": random.randint(300, 2000),
        "B": random.randint(300, 2000),
        "C": random.randint(300, 2000),
        # D: "végtelen"
        "D": 10**9
    }

    # Igények: 4000 szétosztva 6 célra, mind >= 200
    demand_list = random_demands(total=4000, n=len(destinations), min_each=200)
    demands: Dict[str, int] = {d: demand_list[i] for i, d in enumerate(destinations)}

    #LP modell
    prob = pl.LpProblem("Transport problem", pl.LpMinimize)

    x_keys = [(w, d) for w in warehouses for d in destinations]
    x = pl.LpVariable.dicts("x", x_keys, lowBound=0, cat=pl.LpContinuous)

    # Célfüggvény: összköltség minimalizálása
    prob += pl.lpSum(x[(w, d)] * costs[w][d] for w, d in x_keys)

    # Kínálati korlátok: sum_j x_{w,j} ≤ supply_w
    for w in warehouses:
        prob += pl.lpSum(x[(w, d)] for d in destinations) <= supplies[w], f"Supply_{w}"

    # Keresleti egyenlőségek: sum_i x_{i,d} = demand_d
    for d in destinations:
        prob += pl.lpSum(x[(w, d)] for w in warehouses) == demands[d], f"Demand_{d}"

    # Megoldás
    solver = pl.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        print(f"FIGYELEM: megoldási státusz = {status}. Lehet, hogy nincs optimális megoldás.")
    total_cost = pl.value(prob.objective)

    # Kimenet: szállítási mátrix (honnan-hova mennyit viszünk)
    ship_matrix = []
    for w in warehouses:
        row = []
        for d in destinations:
            val = x[(w, d)].value() or 0.0
            row.append(val)
        ship_matrix.append(row)

    # költségmátrix és készlet/igény
    if pd is not None:
        cost_df = pd.DataFrame([[costs[w][d] for d in destinations] for w in warehouses],
                               index=warehouses, columns=destinations)
        print("Véletlen egységköltségek (D minden oszlopban a legdrágább):")
        print(cost_df.to_string())
        print()

        meta_df = pd.DataFrame({
            "Készlet": [supplies[w] if w != "D" else "∞" for w in warehouses]
        }, index=warehouses)
        dem_df = pd.DataFrame({"Igény": [demands[d] for d in destinations]}, index=destinations)
        print("Készletek:")
        print(meta_df.to_string())
        print()
        print("Igények:")
        print(dem_df.to_string())
        print()
    else:
        print("Költségek (Ft/egység):")
        print_table([[costs[w][d] for d in destinations] for w in warehouses],
                    warehouses, destinations, decimals=0)

        print("Készletek:")
        for w in warehouses:
            print(f"  {w}: {'∞' if w=='D' else supplies[w]}")
        print("\nIgények:")
        for d in destinations:
            print(f"  {d}: {demands[d]}")
        print()

    print(f"Megoldási státusz: {status}")
    print(f"Minimális összköltség: {total_cost:.2f}\n")

    print_table(ship_matrix, warehouses, destinations,
                title="Optimális szállítási mátrix (egység):", decimals=2)


if __name__ == "__main__":
    main()