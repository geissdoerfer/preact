import enmanage
import numpy as np
import pytest
from datetime import datetime
from enmanage.prediction import *
from nasadata import fxt_dataset


def test_prediction(en_predictor, fxt_dataset):

    doy = np.array(fxt_dataset['data'].index.dayofyear)[:365]
    e_in = fxt_dataset['data']['exposure'].as_matrix()[:365]

    mses = np.zeros(len(doy))
    for i in range(len(doy)):
        en_predictor.step(doy[i], e_in[i])
        e_in_pred = en_predictor.predict(doy)
        mses[i] = np.mean((e_in - e_in_pred) ** 2)

    assert(mses[-1] < 15.0)


@pytest.fixture(params=[EWMA, AST, MBSGD, OPTMODEL])
def en_predictor(fxt_dataset, request):

    doy = np.array(fxt_dataset['data'].index.dayofyear)
    e_in = fxt_dataset['data']['exposure'].as_matrix()

    if(request.param is AST):
        predictor = request.param(
            doy[-365:],
            e_in[-365:],
            fxt_dataset['latitude']
            )
    elif(request.param is OPTMODEL):
        predictor = request.param(e_in[:2*365])
    else:
        predictor = request.param()

    return predictor
