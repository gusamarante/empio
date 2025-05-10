import os
import pandas as pd
from xlogit.utils import wide_to_long


def load_restaurant(data_format="wide"):
    """
    Simulated data on the restaurant choices of 300 families
    """
    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, "restaurant_ps1.csv"))

    if data_format == "wide":
        return df
    elif data_format == "long":
        df = wide_to_long(
            dataframe=df,
            id_col="family.id",
            alt_list=[
                'CafeEccell',
                'Christophers',
                'Freebirds',
                'LosNortenos',
                'MadCows',
                'MamasPizza',
                'WingsNmore',
            ],
            varying=[
                'cost',
                'distance',
            ],
            alt_name='restaurant',
            sep='.',
        )
        return df
    else:
        raise TypeError("Invalid data format")
