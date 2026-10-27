def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    return (x - a) / (b - a) if x <= b else (c - x) / (c - b)

def temp_sets(t):
    return {
        'cold': tri(t, -1, 0, 20),
        'warm': tri(t, 15, 25, 35),
        'hot': tri(t, 30, 40, 41)
    }

def hum_sets(h):
    return {
        'dry': tri(h, -1, 0, 50),
        'humid': tri(h, 40, 100, 101)
    }

speed_sets = {
    'low': (0, 20, 50),
    'medium': (30, 50, 70),
    'high': (50, 80, 100)
}

rules = [
    ('cold', 'dry', 'low'),
    ('cold', 'humid', 'low'),
    ('warm', 'dry', 'medium'),
    ('warm', 'humid', 'high'),
    ('hot', 'dry', 'high'),
    ('hot', 'humid', 'high')
]

def fan_speed(t, h):
    T, H = temp_sets(t), hum_sets(h)
    strength = {}

    for tt, hh, out in rules:
        w = min(T[tt], H[hh])
        strength[out] = max(strength.get(out, 0), w)

    num = den = 0.0

    for x in range(101):
        mu = max(
            min(strength.get(o, 0), tri(x, *p))
            for o, p in speed_sets.items()
        )
        num += x * mu
        den += mu

    return T, H, strength, (num / den if den else 0)

for t, h in [(10, 30), (28, 60), (38, 80)]:
    T, H, s, y = fan_speed(t, h)

    print(f"Temp={t}C Hum={h}%")
    print("Temp memberships:", {k: round(v, 2) for k, v in T.items()})
    print("Humidity memberships:", {k: round(v, 2) for k, v in H.items()})
    print("Rule strengths:", {k: round(v, 2) for k, v in s.items()})
    print(f"Fan speed (centroid) = {y:.2f}%\n")
