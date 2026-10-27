rules = {
    'battery_fault': [
        ['screen_black', 'power_led_off', 'charger_connected']
    ],
    'display_fault': [
        ['screen_black', 'power_led_on', 'fan_running']
    ],
    'ram_fault': [
        ['power_led_on', 'beeps']
    ]
}

facts = {
    'screen_black',
    'power_led_off',
    'charger_connected'
}

def bc(goal, depth=0):
    pad = ' ' * depth

    if goal in facts:
        print(f"{pad}{goal}: found in facts -> True")
        return True

    if goal not in rules:
        print(f"{pad}{goal}: no fact, no rule -> False")
        return False

    for prem in rules[goal]:
        print(f"{pad}Goal {goal}: need {prem}")

        if all(bc(p, depth + 1) for p in prem):
            print(f"{pad}{goal} -> True")
            return True

    print(f"{pad}{goal} -> False")
    return False

for g in ['battery_fault', 'display_fault', 'ram_fault']:
    print(f"\n== Proving {g} ==")
    print("Result:", bc(g))
