import os
import pandas as pd
from xlogit.utils import wide_to_long

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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


def load_commute(data_format="wide"):
    """
    commute_binary.csv variables

    id: individual identifier
    mode: commute mode
    time.car: time (in minutes) to drive to campus
    cost.car: cost (in dollars) to drive to campus
    time.bus: time (in minutes) to ride the bus to campus
    cost.bus: cost (in dollars) to ride the bus to campus
    price_gas: gasoline price ($ per gallon)
    snowfall: snowfall (in inches) over previous 24 hours
    construction: binary indicator (0 or 1) for road construction on route
    bus_detour: binary indicator (0 or 1) if bus is on a detour
    age: age (in years)
    income: annual income (in 1000 dollars)
    marital_status: marital status, either "single" or "married"
    """
    base_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_dir, "commute_binary.csv"))
    df = df.rename({"mode": "choice"}, axis=1)

    if data_format == "wide":
        return df
    elif data_format == "long":
        df = wide_to_long(
            dataframe=df,
            id_col="id",
            alt_list=[
                'bus',
                'car',
            ],
            varying=[
                'cost',
                'time',
            ],
            alt_name='mode',
            sep='.',
        )
        return df
    else:
        raise TypeError("Invalid data format")
