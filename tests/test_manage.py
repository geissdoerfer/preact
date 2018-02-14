import numpy as np
import pytest
import enmanage
from enmanage.managers import *
from nasadata import fxt_dataset, eval_data, training_data


def eval_budget(doy, e_in, budget):
    rup = enmanage.relative_underperformance(
        e_in, budget, enmanage.profiles['uniform'](doy))

    efficiency = np.mean(budget)/np.mean(e_in)

    assert(rup < 1.0)
    assert(efficiency > 0.0)


def test_lteno(eval_data, simulator, training_data, fxt_dataset):

    predictor = enmanage.AST(
        training_data['doy'], training_data['e_in'], fxt_dataset['latitude'])

    en_manager = enmanage.LTENO(
        max(eval_data['e_in']),
        simulator.battery.capacity,
        predictor.d,
        simulator.battery.get_eta_in(),
        simulator.battery.get_eta_out()
    )

    budget = np.zeros(len(eval_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(eval_data['doy'], eval_data['e_in'])):

        e_in_real, budget[i], _ = simulator.step(doy, e_in)

        predictor.step(doy, e_in_real)
        e_pred = predictor.predict(np.arange(365))

        duty_cycle = en_manager.calc_duty_cycle(e_pred, predictor.alpha)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(eval_data['doy'], eval_data['e_in'], budget)


def test_preact(eval_data, simulator, predictor):

    en_manager = enmanage.PREACT(
        simulator.battery.capacity,
        enmanage.profiles['uniform']
    )

    budget = np.zeros(len(eval_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(eval_data['doy'], eval_data['e_in'])):

        e_in_real, budget[i], soc = simulator.step(doy, e_in)

        predictor.step(doy, e_in_real)
        e_pred = predictor.predict(np.arange(doy + 1, doy + 1 + 365))

        duty_cycle = en_manager.calc_duty_cycle(doy, soc, e_pred)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(eval_data['doy'], eval_data['e_in'], budget)


def test_stewma(eval_data, simulator):

    predictor = enmanage.EWMA()

    en_manager = enmanage.STEWMA(
        max(eval_data['e_in'])
    )

    budget = np.zeros(len(eval_data['doy']))
    for i, (doy, e_in) in enumerate(
            zip(eval_data['doy'], eval_data['e_in'])):

        e_in_real, budget[i], soc = simulator.step(doy, e_in)

        predictor.step(e_in_real)
        e_pred = predictor.predict()

        duty_cycle = en_manager.calc_duty_cycle(e_pred)
        simulator.set_duty_cycle(duty_cycle)

    eval_budget(eval_data['doy'], eval_data['e_in'], budget)


@pytest.fixture()
def predictor(fxt_dataset):
    e_in = (
        fxt_dataset['data']['exposure'].as_matrix()[365:] * enmanage.PWR_FACTOR
    )
    return enmanage.CLAIRVOYANT(e_in)


@pytest.fixture()
def simulator(eval_data, predictor):
    consumer = enmanage.Consumer(max(eval_data['e_in']))
    battery = enmanage.Battery(3500.0 * enmanage.CAP_FACTOR, 0.5)
    simulator = enmanage.Simulator(consumer, battery)

    return simulator
