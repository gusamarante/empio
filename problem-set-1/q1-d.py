from empio.data import load_restaurant
from xlogit import MultinomialLogit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import getpass


username = getpass.getuser()

# Read data
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
model_output = pd.DataFrame(
    index=vars_x,
    data={
        "Coeff": model.coeff_,
        "Std Error": model.stderr,
        "Z-stat": model.zvalues,
        "p-values": model.pvalues,
    },
)


# =================================
# ===== Compute Probabilities =====
# =================================
# Original
_, probs0 = model.predict(
    X=data[vars_x],
    varnames=vars_x,
    ids=data["family.id"],
    alts=data["restaurant"],
    return_proba=True,
)
probs0 = pd.DataFrame(data=probs0, columns=model.alternatives, index=data['family.id'].unique())
probs0.index.name = "family.id"

# Counterfactual
counter_data = data.copy()
is_chris = counter_data['restaurant'] == 'Christophers'
counter_data['cost'] = (counter_data['cost'] * 0.75) * is_chris + counter_data['cost'] * (~is_chris)

_, probs1 = model.predict(
    X=counter_data[vars_x],
    varnames=vars_x,
    ids=counter_data["family.id"],
    alts=counter_data["restaurant"],
    return_proba=True,
)
probs1 = pd.DataFrame(data=probs1, columns=model.alternatives, index=counter_data['family.id'].unique())
probs1.index.name = "family.id"


print(probs1.sum() - probs0.sum())


# ============================
# ===== Consumer Surplus =====
# ============================
Vni0 = data[vars_x] * model_output['Coeff']
Vni0 = Vni0.set_index([data['family.id'], data['restaurant']]).sum(axis=1)
logsum_obs = np.log(np.exp(Vni0).groupby('family.id').sum())

Vni1 = counter_data[vars_x] * model_output['Coeff']
Vni1 = Vni1.set_index([counter_data['family.id'], counter_data['restaurant']]).sum(axis=1)
logsum_counter = np.log(np.exp(Vni1).groupby('family.id').sum())

# Change in consumer surplus for each family
dCS = (logsum_counter - logsum_obs) / (- model_output.loc['cost', 'Coeff'])

print(dCS.sum())
