prior, sens, spec = 0.02, 0.95, 0.90

def update(p, positive=True):
    like_d = sens if positive else 1 - sens
    like_n = (1 - spec) if positive else spec
    evidence = like_d * p + like_n * (1 - p)
    return like_d * p / evidence, evidence

p1, e1 = update(prior)
p2, e2 = update(p1)
pn, _ = update(prior, positive=False)

print(f"Prior P(Disease) = {prior:.4f}")
print(f"P(Test+) = {e1:.4f}")
print(f"P(Disease | Test1 +) = {p1:.4f}")
print(f"P(Disease | Test1 +, Test2 +) = {p2:.4f}")
print(f"P(Disease | Test1 -) = {pn:.4f}")
