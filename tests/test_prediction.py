import enmanage
import numpy as np
import pandas as pd
import pathlib
import pytest
from datetime import datetime
from enmanage.prediction import Model, mfun, mfun_d_a, mfun_d_b, fit_optimal


def mse(xs, ys, model, parameters):
    return np.mean((ys - model.mfun(xs, parameters)) ** 2)


def test_learning(dataset):

    xs = np.array(dataset.index.dayofyear)[:365]
    ys = dataset['exposure'].as_matrix()[:365]

    predictor = enmanage.MBSGD(scale=1.0)

    model = Model(mfun, (mfun_d_a, mfun_d_b))
    opt_parameters = fit_optimal(xs, ys, model)

    mses = np.zeros(len(xs))
    for i in range(len(xs)):
        predictor.step(xs[i], ys[i])
        mses[i] = mse(xs, ys, model, predictor.params)

    mse_offline = mse(xs, ys, model, opt_parameters)
    mse_online = mse(xs, ys, model, predictor.params)

    assert(abs(mse_offline - mse_online) < 0.25 * mse_offline)


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
