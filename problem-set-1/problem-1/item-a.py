from empio.data import load_restaurant
from xlogit import MultinomialLogit
import pandas as pd

data = load_restaurant(data_format="long")

# add alternative specific constants (asc)
outside_option = "Freebirds"
rest_dummies = pd.get_dummies(data['restaurant'], prefix="asc").astype(int)
data = pd.concat([data, rest_dummies], axis=1)
data = data.drop(f"asc_{outside_option}", axis=1)


# ===========================
# ===== Estimate mlogit =====
# ===========================
vars_x = [
    'asc_CafeEccell',
    'asc_Christophers',
    # 'asc_Freebirds',
    'asc_LosNortenos',
    'asc_MadCows',
    'asc_MamasPizza',
    'asc_WingsNmore',
    'cost',
    'distance',
]

model = MultinomialLogit()
model.fit(
    X=data[vars_x],
    y=data["choice"],
    varnames=vars_x,
    ids=data["family.id"],
    alts=data["restaurant"]
)
print(model.summary())
coeffs = pd.Series(index=vars_x, data=model.coeff_)


# ================================
# ===== Compute elasticities =====
# ================================
# Fitted probabilities
_, probs = model.predict(
    X=data[vars_x],
    varnames=vars_x,
    ids=data["family.id"],
    alts=data["restaurant"],
    return_proba=True,
)
probs = pd.DataFrame(data=probs, columns=model.alternatives)

# marginal effects and elasticities
me = pd.DataFrame(columns=model.alternatives)
el = pd.DataFrame(columns=model.alternatives)
wrt = "Freebirds"
for rest in model.alternatives:

    if rest == wrt:
        me[rest] = 1
    else:
        me[rest] = coeffs.loc['cost']


# TODO report qunatiles of elasticities
# TODO chart of elasticity per alternative

a = 1