from empio.data import load_restaurant
from xlogit import MultinomialLogit
import pandas as pd
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
    alts=data["restaurant"],

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
probs = pd.DataFrame(data=probs, columns=model.alternatives, index=data['family.id'].unique())
probs.index.name = "family.id"

# marginal effects and elasticities
me = pd.DataFrame(columns=model.alternatives)
el = pd.DataFrame(columns=model.alternatives)
wrt = "Freebirds"  # plays the role of index j
wrt_cost = data.set_index(['family.id', 'restaurant']).xs(wrt, level=1)['cost']
for rest in model.alternatives:

    if rest == wrt:
        me[rest] = coeffs.loc['cost'] * probs[wrt] * (1 - probs[wrt])
        el[rest] = coeffs.loc['cost'] * wrt_cost * (1 - probs[wrt])
    else:
        me[rest] = - coeffs.loc['cost'] * probs[wrt] * probs[rest]
        el[rest] = - coeffs.loc['cost'] * wrt_cost * probs[wrt]


# ==============================
# ===== Chart Elasticities =====
# ==============================
df2plot = el[["Freebirds", "CafeEccell"]].stack().reset_index().rename({"level_1": "restaurant", 0: "elasticity"}, axis=1)
df2plot = df2plot[df2plot["restaurant"].isin(('Freebirds', 'CafeEccell'))]

ax = sns.displot(df2plot, x="elasticity", hue="restaurant", kind="kde")
sns.move_legend(ax, "upper center")
plt.tight_layout()
plt.savefig(f'/Users/{username}/Dropbox/PhD/Econometria Estrutural/Problem Set 1/figures/Q1 A - KDE of elasticities.pdf')
plt.show()
