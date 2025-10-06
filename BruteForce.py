import itertools

tasks = [(3, 5), (2, 4), (4, 6)]  # csak az időt használjuk most: (duration, cost)

def sum_completion_times(sequence):
    t = 0
    sumC = 0
    for task in sequence:
        t += task[0]        # aktuális befejezési idő
        sumC += t           # összeadjuk a befejezési időt
    return sumC

best_seq = None
best_score = float("inf")

for perm in itertools.permutations(tasks):
    score = sum_completion_times(perm)
    if score < best_score:
        best_score = score
        best_seq = perm

print("Optimális sorrend (SPT elvhez hasonló):", best_seq)
print("Minimális össz befejezési idő (sum Cj):", best_score)