import numpy as np
import pytest
import enmanage
from enmanage.managers import *
from nasadata import fxt_dataset


def eval_budget(doy, e_in, budget):
    rup = enmanage.relative_underperformance(
        e_in, budget, enmanage.profiles['uniform'](doy))

    efficiency = np.mean(budget)/np.mean(e_in)

    assert(rup < 1.0)
    assert(efficiency > 0.0)


def test_lteno(simulator, sample_data, fxt_dataset):
    doy_training = np.array(fxt_dataset['data'].index.dayofyear)[365:3 * 365]
    e_in_training = (
        fxt_dataset['data']
        ['exposure'].as_matrix()[365:3 * 365] * enmanage.PWR_FACTOR
    )

    predictor = enmanage.AST(
        doy_training, e_in_training, fxt_dataset['latitude'])

    en_manager = enmanage.LTENO(
        max(sample_data['e_in']),
        simulator.battery.capacity,
        predictor.d,
        simulator.battery.get_eta_in(),
        simulator.battery.get_eta_out()
    )

    budget = np.zeros(len(sample_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(sample_data['doy'], sample_data['e_in'])):

        e_in_real, budget[i], _ = simulator.step(doy, e_in)

        predictor.step(doy, e_in_real)
        e_pred = predictor.predict(np.arange(365))

        duty_cycle = en_manager.calc_duty_cycle(e_pred, predictor.alpha)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(sample_data['doy'], sample_data['e_in'], budget)


def test_preact(simulator, predictor, sample_data):

    en_manager = enmanage.PREACT(
        simulator.battery.capacity,
        enmanage.profiles['uniform']
    )

    budget = np.zeros(len(sample_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(sample_data['doy'], sample_data['e_in'])):

        e_in_real, budget[i], soc = simulator.step(doy, e_in)

        predictor.step(doy, e_in_real)
        e_pred = predictor.predict(np.arange(doy + 1, doy + 1 + 365))

        duty_cycle = en_manager.calc_duty_cycle(doy, soc, e_pred)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(sample_data['doy'], sample_data['e_in'], budget)


def test_stewma(simulator, predictor, sample_data):

    en_manager = enmanage.STEWMA(
        max(sample_data['e_in'])
    )

    budget = np.zeros(len(sample_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(sample_data['doy'], sample_data['e_in'])):

        e_in_real, budget[i], soc = simulator.step(doy, e_in)

        predictor.step(doy, e_in_real)
        e_pred = predictor.predict(np.arange(doy + 1, doy + 1 + 365))

        duty_cycle = en_manager.calc_duty_cycle(doy, soc, e_pred)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(sample_data['doy'], sample_data['e_in'], budget)


@pytest.fixture()
def sample_data(fxt_dataset):
    doy = np.array(fxt_dataset['data'].index.dayofyear)[365:3 * 365]
    e_in = (
        fxt_dataset['data']
        ['exposure'].as_matrix()[365:3 * 365] * enmanage.PWR_FACTOR
    )

    return {'doy': doy, 'e_in': e_in}


@pytest.fixture()
def predictor(fxt_dataset):
    e_in = fxt_dataset['data']['exposure'].as_matrix() * enmanage.PWR_FACTOR
    return enmanage.CLAIRVOYANT(e_in)


@pytest.fixture()
def simulator(sample_data, predictor):
    consumer = enmanage.Consumer(max(sample_data['e_in']))
    battery = enmanage.Battery(3500.0 * enmanage.CAP_FACTOR, 0.5)
    simulator = enmanage.Simulator(consumer, battery)

    return simulator
