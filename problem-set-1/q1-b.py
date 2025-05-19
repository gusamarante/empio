from empio.data import load_restaurant
from xlogit import MultinomialLogit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import getpass


username = getpass.getuser()

# Read data
data = load_restaurant(data_format="long")

# Create the new variable
data['cost-income'] = data['cost'] / data['income']

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
    'cost-income',
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
# model_output.to_clipboard()

# ====================
# ===== Analysis =====
# ====================
analysis = pd.DataFrame(
    index=[f"family {i + 1}" for i in range(3)],
    data={"income": [20, 40, 60]},
)
# marginal utility of income
analysis['mui'] = - model_output.loc["cost-income", "Coeff"] / analysis['income']

# dollar value of reduction in distance
analysis['dvrd'] = (model_output.loc["distance", "Coeff"] / model_output.loc["cost-income", "Coeff"]) * analysis['income']

print(analysis)