from empio.data import load_restaurant
from xlogit import MultinomialLogit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import getpass
from tqdm import tqdm


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
model_output.to_clipboard()

# ====================
# ===== Analysis =====
# ====================
analysis = pd.DataFrame(
    index=[f"family {i + 1}" for i in range(3)],
    data={"income": [20, 40, 60]},
)
# marginal utility of income as a reduction in cost
analysis['mui'] = - model_output.loc["cost-income", "Coeff"] / analysis['income']


# marginal utility of income as the chosen restaurant
mui_rp = pd.DataFrame()
for fid in tqdm(data['family.id'].unique()):
    choice = data[data['family.id'] == fid]['choice'].iloc[0]
    znj = data[(data['family.id'] == fid) & (data['restaurant'] == choice)].iloc[0]

    mui_rp.loc[fid, 'income'] = znj.loc['income']
    mui_rp.loc[fid, 'mui'] = - model_output.loc["cost-income", "Coeff"] * znj.loc['cost'] / (znj.loc['income']**2)


mui_rp = mui_rp.sort_values('income')
grouped = mui_rp.groupby('income').mean()


size = 5
fig = plt.figure(figsize=(size * (16 / 7.3), size))
ax = plt.subplot2grid((1, 1), (0, 0))
ax.scatter(mui_rp['income'], mui_rp['mui'], label='Family', edgecolor=None)
ax.set_xlabel("Income (thousands, log scale)")
ax.set_ylabel("Marginal Utility of Income (log scale)")
ax.set_yscale('log')
ax.set_xscale('log')
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.savefig(f'/Users/{username}/Dropbox/PhD/Econometria Estrutural/Problem Set 1/figures/Q2 C - loglog income mui.pdf')
plt.show()
plt.close()


# dollar value of reduction in distance
analysis['dvrd'] = (model_output.loc["distance", "Coeff"] / model_output.loc["cost-income", "Coeff"]) * analysis['income']

print(analysis)