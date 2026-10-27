rules = [
    ({'employed', 'salary_above_30k'}, 'stable_income'),
    ({'stable_income', 'credit_score_good'}, 'low_risk'),
    ({'low_risk', 'has_collateral'}, 'loan_approved'),
    ({'loan_approved'}, 'send_offer_letter')
]

facts = {
    'employed',
    'salary_above_30k',
    'credit_score_good',
    'has_collateral'
}

print("Initial facts:", sorted(facts))

fired = True
n = 0

while fired:
    fired = False
    for prem, concl in rules:
        if prem <= facts and concl not in facts:
            facts.add(concl)
            n += 1
            fired = True
            print(f"Cycle {n}: {sorted(prem)} => {concl}")

print("Final facts:", sorted(facts))
print("Loan approved?", 'loan_approved' in facts)
