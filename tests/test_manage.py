import enmanage
import numpy as np
import pandas as pd
import pathlib
import pytest
from datetime import datetime
import matplotlib.pyplot as plt


def test_manage(dataset):

    xs = np.array(dataset.index.dayofyear)[:-365]
    ys = dataset['exposure'].as_matrix() * enmanage.PWR_FACTOR

    en_predictor = enmanage.OPTMODEL(ys, scale=enmanage.PWR_FACTOR)
    battery = enmanage.Battery(3500.0 * enmanage.CAP_FACTOR, 0.5)
    en_manager = enmanage.PREACT(
        battery.capacity, enmanage.profiles['uniform'],
        {'k_p': 0.25, 'k_i': 0.0, 'k_d': 0.0})

    simulator = enmanage.Simulator(en_manager, en_predictor, battery)

    budget = np.zeros(len(xs))
    for i in range(len(xs)):
        budget[i] = simulator.step(xs[i], ys[i])

    rup = enmanage.relative_underperformance(
        ys[:len(xs)], budget, enmanage.profiles['uniform'](xs))

    efficiency = np.mean(budget)/np.mean(ys[:len(xs)])

    assert(rup < 0.25)
    assert(efficiency > 0.9)


@pytest.fixture(params=['A_MRA', 'B_ASP', 'C_BLN', 'D_HBN', 'E_IJK'])
def dataset(request):
    filepath = (
        pathlib.Path(__file__).absolute().parents[1]
        / 'data'
        / 'energy'
        / f'{ request.param }.txt'
    )

    df = pd.read_csv(
        filepath, delim_whitespace=True, skiprows=8, parse_dates=[[0, 1, 2]],
        date_parser=lambda *columns: datetime(*map(int, columns)))

    df.columns = ['date', 'exposure']
    df.set_index(df['date'], inplace=True)
    df.drop('date', 1, inplace=True)

    return df


class FakeRequest(object):
    def __init__(self, param):
        self.param = param
