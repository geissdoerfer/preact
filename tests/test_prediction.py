import enmanage
import numpy as np
import pandas as pd
import pathlib
import pytest
import parse
from datetime import datetime
from enmanage.prediction import *


def mse(doy, e_in_truth, e_in_pred):
    return np.mean((ys - model.mfun(xs, parameters)) ** 2)


def test_prediction(meta_fix):

    en_predictor, doy, e_in = meta_fix

    mses = np.zeros(len(doy))
    for i in range(len(doy)):
        en_predictor.step(doy[i], e_in[i])
        e_in_pred = en_predictor.predict(doy)
        mses[i] = np.mean((e_in - e_in_pred) ** 2)

    assert(mses[-1] < 15.0)


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

    with open(filepath, 'r') as f:
        for _ in range(3):
            line = f.readline()
        latitude = parse.search('Latitude {:f}', line)[0] / 360.0 * 2 * np.pi

    return {'data': df, 'latitude': latitude}


@pytest.fixture(params=[EWMA, AST, MBSGD, OPTMODEL])
def meta_fix(dataset, request):

    doy = np.array(dataset['data'].index.dayofyear)
    e_in = dataset['data']['exposure'].as_matrix()

    if(request.param is AST):
        en_predictor = request.param(
            doy[:365],
            e_in[:365],
            dataset['latitude']
            )
    elif(request.param is OPTMODEL):
        en_predictor = request.param(e_in[365:3*365])
    else:
        en_predictor = request.param()

    return en_predictor, doy[365:2*365], e_in[365:2*365]
