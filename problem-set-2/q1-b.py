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
        inst = self.instrument

        # Difference in representative utility
        # V_car - V_bus
        util_dif = alpha + beta * exog[:, 1] + g_car * exog[:, 2] - g_bus * exog[:, 3]

        prob = 1 / (1 + np.exp(- util_dif))

        resid = endog - prob

        # Moment matrix
        moms = inst.mul(pd.Series(resid), axis=0)

        return moms

var_cols = ['const', 'cost.car', 'time.car', 'time.bus']
inst_cols = ['const', 'price_gas', 'snowfall', 'construction', 'bus_detour']
p0 = np.array([0, 0, 0, 0])
bl = BinaryLogit(
    endog=data['asc_car'],
    exog=data[var_cols],
    instrument=data[inst_cols],
).fit(p0)
print(bl.summary())


table_out = pd.DataFrame(
    index=var_cols,
    data={
        'Param': bl.params,
        'Std. Err.': bl.bse,
        't-stat': bl.tvalues,
        'p-value': bl.pvalues,
    },
)
table_out.to_clipboard()
