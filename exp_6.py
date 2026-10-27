facts = {"wifi_connected", "router_on"}

rules = [
    ({"wifi_connected", "router_on"}, "check_internet"),
    ({"check_internet"}, "internet_available")
]

changed = True

while changed:
    changed = False
    for condition, result in rules:
        if condition.issubset(facts) and result not in facts:
            facts.add(result)
            print("Derived:", result)
            changed = True

print("Final facts:", facts)
