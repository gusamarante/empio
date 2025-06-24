# TODO match random numbers with manu
# TODO run the canned version of the model

import numpy as np
from empio.data import load_camping


np.random.seed(703)

data = load_camping()

n_draws = 10  # number of random draws  # TODO make this big
n_rand_coeff = 2  # number of random coefficients
n_ids = data['camper_id'].max()  # number of individuals

coeff_draws = n_ids * n_draws  # per parameter

draws = np.random.normal(size=(coeff_draws, n_rand_coeff))

def choice_prob(params):
    # TODO this is for a single camper

    betac = params[0]
    mut = params[0]
    sig2t = params[0]
    mum = params[0]
    sig2m = params[0]

    # TODO transform the standard normal draws into with their parameters
    # TODO compute representative utility for every alternative for every draw
    # TODO Calculate the conditional choice probability for every alternative for each draw
    # TODO Calculate the simulated choice probability for every alternative as the mean over all draws

def sloglike(params):

    # TODO Get the choice_prob for each camper
    # TODO Sum the log of the simulated choice probability for each camper’s chosen alternative.
    # TODO Return the negative of the log of simulated likelihood
     pass
