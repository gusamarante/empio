from empio.data import load_commute
import pandas as pd
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM

data = load_commute(data_format="wide")

# add alternative specific constants (asc)
outside_option = "bus"
dummies = pd.get_dummies(data['choice'], prefix="asc").astype(int)
data = pd.concat([data, dummies], axis=1)
data = data.drop(f"asc_{outside_option}", axis=1)
data['const'] = 1


class BinaryLogit(GMM):

    def momcond(self, params):
        alpha = params[0]
        beta = params[1]
        g_car = params[2]
        g_bus = params[3]

        endog = self.endog
        exog = self.exog

        # Difference in representative utility
        # V_car - V_bus
        util_dif = alpha + beta * data['cost.car'] + g_car * data['time.car'] - g_bus * data['time.bus']

        prob = 1 / (1 + np.exp(- util_dif))

        resid = data['asc_car'] - prob

        # Moment matrix
        X = pd.concat([pd.Series(np.ones(resid.shape[0]), name='cons'), data[['cost.car', 'time.car', 'time.bus']]], axis=1)
        moms = X.mul(resid, axis=0)

        return moms


p0 = np.array([0, 0, 0, 0])
bl = BinaryLogit(
    endog=data['asc_car'],
    exog=data[['const', 'cost.car', 'time.car', 'time.bus']],
    instrument=data[['const', 'cost.car', 'time.car', 'time.bus']],
).fit(p0)
print(bl.summary())
