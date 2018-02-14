import enmanage
import numpy as np
import pytest
from datetime import datetime
from enmanage.prediction import *
from nasadata import fxt_dataset, eval_data, training_data


def test_mbsgd(eval_data):
    predictor = enmanage.MBSGD()
    eval_predictor(predictor, eval_data['doy'], eval_data['e_in'])


def test_sgd(eval_data):
    predictor = enmanage.SGD()
    eval_predictor(predictor, eval_data['doy'], eval_data['e_in'])


def test_ast(eval_data, training_data, fxt_dataset):
    predictor = enmanage.AST(
        training_data['doy'], training_data['e_in'], fxt_dataset['latitude'])
    eval_predictor(predictor, eval_data['doy'], eval_data['e_in'])


def eval_predictor(en_predictor, doys, e_ins):
    mses = np.zeros(len(doys))
    for i, (doy, e_in) in enumerate(zip(doys, e_ins)):
        en_predictor.step(doy, e_in)
        e_in_pred = en_predictor.predict(np.arange(doy + 1, doy + 1 + 365))
        mses[i] = np.mean((e_in - e_in_pred) ** 2)

    assert(mses[-1] < 50.0)


def test_ewma(eval_data):
    en_predictor = enmanage.EWMA()

    errors = np.zeros(len(eval_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(eval_data['doy'], eval_data['e_in'])):
        en_predictor.step(e_in)
        e_in_pred = en_predictor.predict()
        errors[i] = eval_data['e_in'][i] - e_in_pred

    assert(np.mean(errors ** 2) < 15.0)
