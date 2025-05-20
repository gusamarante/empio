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

# Add a dummy for 2 groups of number of kids: (1) zero or one kids, (2) two or more
data['kids2m'] = (data['kids'] >= 2).astype(int)


# ================================
# ===== Model with less kids =====
# ================================
aux_data = data[data['kids2m'] == 0]
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
    X=aux_data[vars_x],
    y=aux_data["choice"],
    varnames=vars_x,
    ids=aux_data["family.id"],
    alts=aux_data["restaurant"],
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


# ================================
# ===== Model with less kids =====
# ================================
aux_data = data[data['kids2m'] == 1]
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
    X=aux_data[vars_x],
    y=aux_data["choice"],
    varnames=vars_x,
    ids=aux_data["family.id"],
    alts=aux_data["restaurant"],
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


# ========================================
# ===== Model with dummy interaction =====
# ========================================
aux_data = data.copy()
aux_data['kids2m x cost'] = aux_data["kids2m"] * aux_data["cost"]
aux_data['kids2m x dist'] = aux_data["kids2m"] * aux_data["distance"]

vars_x = [
    'asc_CafeEccell',
    'asc_Christophers',
    # 'asc_Freebirds',
    'asc_LosNortenos',
    'asc_MadCows',
    'asc_MamasPizza',
    'asc_WingsNmore',
    'cost',
    'kids2m x cost',
    'distance',
    'kids2m x dist',
]
model = MultinomialLogit()
model.fit(
    X=aux_data[vars_x],
    y=aux_data["choice"],
    varnames=vars_x,
    ids=aux_data["family.id"],
    alts=aux_data["restaurant"],
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



# =================================================
# ===== Model with number of kids interaction =====
# =================================================
aux_data = data.copy()
aux_data['kids x cost'] = aux_data["kids"] * aux_data["cost"]
aux_data['kids x dist'] = aux_data["kids"] * aux_data["distance"]

vars_x = [
    'asc_CafeEccell',
    'asc_Christophers',
    # 'asc_Freebirds',
    'asc_LosNortenos',
    'asc_MadCows',
    'asc_MamasPizza',
    'asc_WingsNmore',
    'cost',
    'kids x cost',
    'distance',
    'kids x dist',
]
model = MultinomialLogit()
model.fit(
    X=aux_data[vars_x],
    y=aux_data["choice"],
    varnames=vars_x,
    ids=aux_data["family.id"],
    alts=aux_data["restaurant"],
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
