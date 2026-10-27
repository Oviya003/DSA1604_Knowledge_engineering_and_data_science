rules = [
    ('Power', 'RouterOn'),
    ('RouterOn', 'WifiOn'),
    ('WifiOn', 'InternetOK')
]

def infer(known):
    changed = True
    while changed:
        changed = False
        for a, c in rules:
            if known.get(a) is True and c not in known:
                known[c] = True
                changed = True
                print(f"Modus Ponens: {a} -> {c} = True")

            if known.get(c) is False and a not in known:
                known[a] = False
                changed = True
                print(f"Modus Tollens: {c} -> {a} = False")
    return known

print("Case A: Power = True")
print("Result:", infer({'Power': True}))

print("\nCase B: InternetOK = False")
print("Result:", infer({'InternetOK': False}))
