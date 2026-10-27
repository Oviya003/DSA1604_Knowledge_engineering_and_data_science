from itertools import combinations

def neg(l):
    return l[1:] if l.startswith('~') else '~' + l

def resolve(c1, c2):
    out = []
    for l in c1:
        if neg(l) in c2:
            out.append(frozenset((c1 - {l}) | (c2 - {neg(l)})))
    return out

def fmt(c):
    return '{' + ', '.join(sorted(c)) + '}' if c else '{} (empty)'

clauses = [
    frozenset({'~P', 'Q'}),
    frozenset({'~Q', 'R'}),
    frozenset({'P'}),
    frozenset({'~R'})
]

print("Initial clauses:")
for i, c in enumerate(clauses, 1):
    print(f"C{i}: {fmt(c)}")

known = set(clauses)
proved = False
step = 0

while not proved:
    new = set()
    for c1, c2 in combinations(list(known), 2):
        for r in resolve(c1, c2):
            if r not in known and r not in new:
                step += 1
                print(f"Step {step}: resolve {fmt(c1)} & {fmt(c2)} -> {fmt(r)}")
                new.add(r)
                if not r:
                    proved = True
                    break
        if proved:
            break
    if not new:
        break
    known |= new

print("Query R proved by refutation:", proved)
