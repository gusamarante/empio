import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from empio.data import load_restaurant


def logit_loglikelihood(params, x, y, ids):

    exp_vni = np.exp((params * x).sum(axis=1))
    dta = pd.concat([ids.rename("id"), exp_vni.rename('exp_vni')], axis=1)
    sum_exp_vni = dta.groupby('id').sum()['exp_vni']
    dta['sum_exp_vni'] = dta['id'].map(sum_exp_vni)

    probs = dta['exp_vni'] / dta['sum_exp_vni']
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


# ===================
# ===== Example =====
# ===================
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

betas0 = pd.Series(
    data=0.1 * np.ones(len(vars_x)),
    index=vars_x,
)

ll_example = logit_loglikelihood(
    params=betas0,
    x=X,
    y=choices,
    ids=data['family.id'],
)


# ========================
# ===== Optimization =====
# ========================
# Irrestricted Model
vars_x_irr = [
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
X_irr = data[vars_x_irr].copy()
choices_irr = (data['choice'] == data['restaurant']).astype(int)

betas0_irr = pd.Series(
    data=0.1 * np.ones(len(vars_x_irr)),
    index=vars_x_irr,
)

myfunc_irr = lambda b: - logit_loglikelihood(params=b, x=X_irr, y=choices_irr, ids=data['family.id'])

res_irr = minimize(
    fun=myfunc_irr,
    x0=betas0_irr.values,
    method='BFGS',
    options={'disp': True},
)

betas_irr = pd.Series(
    index=vars_x_irr,
    data=res_irr.x,
    name='Irrestricted',
)
ll_irr = - res_irr.fun
std_err_irr = pd.Series(
    data=np.sqrt(np.diag(res_irr.hess_inv)),
    index=vars_x_irr,
)
zvalues = betas_irr / std_err_irr
pvalues = pd.Series(
    data=2 * (1 - norm.cdf(np.abs(zvalues))),
    index=vars_x_irr,
)


# Restricted Model
vars_x_res = [
    # 'asc_CafeEccell',
    # 'asc_Christophers',
    # 'asc_Freebirds',
    # 'asc_LosNortenos',
    # 'asc_MadCows',
    # 'asc_MamasPizza',
    # 'asc_WingsNmore',
    'cost',
    'distance',
]
X_res = data[vars_x_res].copy()
choices_res = (data['choice'] == data['restaurant']).astype(int)

betas0_res = pd.Series(
    data=0.1 * np.ones(len(vars_x_res)),
    index=vars_x_res,
)

myfunc_res = lambda b: - logit_loglikelihood(params=b, x=X_res, y=choices_res, ids=data['family.id'])

res_res = minimize(
    fun=myfunc_res,
    x0=betas0_res.values,
    method='BFGS',
    options={'disp': True},
)

betas_res = pd.Series(
    index=vars_x_res,
    data=res_res.x,
    name="Restricted",
)
ll_res = - res_res.fun

# Put results together
betas = pd.concat([betas_irr, betas_res], axis=1)
ll = pd.Series({"Irrestricted": ll_irr, "Restricted": ll_res})

print(betas)
print(ll)


# ===================
# ===== LR Test =====
# ===================
lr_stat = - 2 * (ll_res - ll_irr)
pval_lr = 1 - chi2.cdf(lr_stat, df=len(vars_x_irr) - len(vars_x_res))
print(f"df={len(vars_x_irr) - len(vars_x_res)}")
print(pval_lr)
