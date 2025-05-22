import numpy as np
import pandas as pd
from empio.data import load_restaurant


def logit_loglikelihood(params, x, y, ids):

    exp_vni = np.exp((params * x).sum(axis=1))
    data = pd.concat([ids.rename("id"), exp_vni.rename('exp_vni')], axis=1)
    tot_exp_vni = data.groupby('id').sum()['exp_vni']
    data['tot_exp_vni'] = data['id'].map(tot_exp_vni)

    probs = data['exp_vni'] / data['tot_exp_vni']
    loglike = (y * np.log(probs)).sum()

    return loglike


# ================
# ===== DATA =====
# ================
data = load_restaurant(data_format='long')

# add alternative specific constants (asc)
outside_option = "Freebirds"
rest_dummies = pd.get_dummies(data['restaurant'], prefix="asc").astype(int)
data = pd.concat([data, rest_dummies], axis=1)
data = data.drop(f"asc_{outside_option}", axis=1)


# =========================
# ===== Organize Data =====
# =========================
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
X = data[vars_x].copy()
choices = (data['choice'] == data['restaurant']).astype(int)

betas = pd.Series(
    data=0.1 * np.ones(len(vars_x)),
    index=vars_x,
)

ll = logit_loglikelihood(
    params=betas,
    x=X,
    y=choices,
    ids=data['family.id'],
)

print(ll)