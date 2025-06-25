# TODO match random numbers with manu
# TODO run the canned version of the model

import numpy as np
import pandas as pd
from tqdm import tqdm
from empio.data import load_camping


np.random.seed(703)

camp_data = load_camping()

n_draws = 2  # number of random draws  # TODO make this big
n_rand_coeff = 2  # number of random coefficients
n_ids = camp_data['camper_id'].max()  # number of individuals

coeff_draws = n_ids * n_draws  # per parameter
draws = np.random.normal(size=(coeff_draws, n_rand_coeff))
draws = pd.DataFrame(
    {
        'betac': 1,
        'betat': draws[:, 0],
        'betam': draws[:, 1],
    },
    index=pd.MultiIndex.from_product(
        [range(1, n_ids + 1), range(1, n_draws + 1)],
        names=['camper_id', 'beta_id'],
    ),
)


def sloglike(params, data):

    betac = params[0]
    mut = params[1]
    sig2t = params[2]
    mum = params[3]
    sig2m = params[4]

    # transform the standard normal draws into with their parameters
    betas = draws.copy()
    betas['betac'] = betas['betac'] * betac
    betas['betat'] = betas['betat'] * np.sqrt(sig2t) + mut
    betas['betam'] = betas['betam'] * np.sqrt(sig2m) + mum

    results = pd.DataFrame(
        columns=data['park'].unique(),
        index=draws.index,
    )
    for cid in tqdm(data['camper_id'].unique()):

        aux_data = data[data['camper_id'] == cid]

        for bid in betas.loc[cid].index:
            p = np.exp((betas.loc[(cid, bid), 'betac'] * aux_data['cost'] + betas.loc[(cid, bid), 'betat'] * aux_data['time'] + betas.loc[(cid, bid), 'betam'] * aux_data['mountain']).values)
            p = pd.Series(
                data = p / p.sum(),
                index=aux_data['park'],
            )
            results.loc[(cid, bid)] = p

    choice_probs = results.groupby('camper_id').mean().astype(float)
    actual_choices = data.pivot(index='camper_id', columns='park', values='choice')[choice_probs.columns]

    sll = (np.log(choice_probs) * actual_choices).sum().sum()
    return - sll

theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
res = sloglike(theta, camp_data)
print(res)

