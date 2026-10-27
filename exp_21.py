import itertools

P_B, P_E = 0.001, 0.002

P_A = {
    (True, True): 0.95,
    (True, False): 0.94,
    (False, True): 0.29,
    (False, False): 0.001
}

P_J = {True: 0.90, False: 0.05}
P_M = {True: 0.70, False: 0.01}

def joint(b, e, a, j, m):
    p = (P_B if b else 1 - P_B) * (P_E if e else 1 - P_E)
    p *= P_A[(b, e)] if a else 1 - P_A[(b, e)]
    p *= P_J[a] if j else 1 - P_J[a]
    p *= P_M[a] if m else 1 - P_M[a]
    return p

def query(var, evidence):
    names = ['B', 'E', 'A', 'J', 'M']
    dist = {}

    for val in (True, False):
        total = 0

        for combo in itertools.product([True, False], repeat=5):
            w = dict(zip(names, combo))

            if w[var] != val:
                continue
            if any(w[k] != v for k, v in evidence.items()):
                continue

            total += joint(*combo)

        dist[val] = total

    z = sum(dist.values())
    return {k: v / z for k, v in dist.items()}

for var, ev in [
    ('B', {'J': True, 'M': True}),
    ('B', {'J': True}),
    ('E', {'J': True, 'M': True}),
    ('A', {'J': True, 'M': True})
]:
    r = query(var, ev)
    print(f"P({var}=T | {ev}) = {r[True]:.4f}")
    print(f"P({var}=F | {ev}) = {r[False]:.4f}")
