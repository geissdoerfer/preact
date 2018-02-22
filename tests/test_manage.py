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


def test_manager(request, manager, consumer, battery, eval_data):

    simulator = enmanage.Simulator(
        manager, manager.predictor, consumer, battery)

    soc, budget, duty_cycle = simulator.run(
        eval_data['doy'], eval_data['e_in'])

    eval_budget(eval_data['doy'], eval_data['e_in'], budget)


@pytest.fixture(params=['LT-ENO', 'PREACT', 'ST-EWMA'])
def manager(request, battery, consumer, training_data, fxt_dataset):
    if(request.param == 'LT-ENO'):
        predictor = enmanage.AST(
            training_data['doy'],
            training_data['e_in'],
            fxt_dataset['latitude'])

        en_manager = enmanage.LTENO(
            predictor,
            consumer,
            battery.capacity,
            battery.get_eta_in(),
            battery.get_eta_out()
        )
    elif(request.param == 'PREACT'):
        e_in = (
            fxt_dataset['data']['exposure'].as_matrix()[365:]
            * enmanage.PWR_FACTOR
        )
        predictor = enmanage.CLAIRVOYANT(e_in)

        en_manager = enmanage.PREACT(
            predictor,
            battery.capacity,
            enmanage.profiles['uniform']
        )

    elif(request.param == 'ST-EWMA'):
        predictor = enmanage.EWMA()

        en_manager = enmanage.STEWMA(
            predictor,
            consumer
        )

    return en_manager


@pytest.fixture
def consumer(eval_data):
    return enmanage.Consumer(max(eval_data['e_in']))


@pytest.fixture
def battery(consumer):
    battery = enmanage.Battery(3500.0 * enmanage.CAP_FACTOR, 0.5)
    return battery
