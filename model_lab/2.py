
print("MODUS PONENS")

pump = True

if pump:
    tank = True

if tank:
    taps = True

if taps:
    garden = True

print("Pump ON:", pump)
print("Tank Filling:", tank)
print("Taps Have Water:", taps)
print("Garden Watered:", garden)

print("\nMODUS TOLLENS")

garden = False

if not garden:
    taps = False

if not taps:
    tank = False

if not tank:
    pump = False

print("Garden Watered:", garden)
print("Taps Have Water:", taps)
print("Tank Filling:", tank)
print("Pump ON:", pump)
