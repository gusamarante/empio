import numpy as np
import pandas as pd
from empio.data import load_camping
from xlogit import MixedLogit


camp_data = load_camping()

choice_name = camp_data[camp_data['choice'] == 1][['camper_id', 'park']].set_index('camper_id').to_dict()
camp_data['choice name'] = camp_data['camper_id'].map(choice_name['park'])

varnames = ['cost', 'time', 'mountain']
model = MixedLogit()
model.fit(
    X=camp_data[varnames],
    y=camp_data['choice name'],
    varnames=varnames,
    alts=camp_data['park'],
    ids=camp_data['camper_id'],
    fit_intercept=False,
    n_draws=1000,  # TODO make this big
    verbose=2,
    randvars={
        'time': 'n',
        'mountain': 'n',
    },
    optim_method='BFGS',
)
print(model.summary())