import numpy as np
import pytest
import enmanage
from enmanage.managers import *
from nasadata import fxt_dataset


def test_manage(fxt_dataset, battery, en_manager):

    doy = np.array(fxt_dataset['data'].index.dayofyear)[:365]
    e_in = fxt_dataset['data']['exposure'].as_matrix() * enmanage.PWR_FACTOR

    en_predictor = enmanage.CLAIRVOYANT(e_in)
    consumer = enmanage.Consumer(max(e_in[:len(doy)]))

    simulator = enmanage.Simulator(consumer, battery)

    budget = np.zeros(len(doy))
    for i in range(len(doy)):
        e_in_real, budget[i], soc = simulator.step(doy[i], e_in[i])
        en_predictor.step(doy[i], e_in_real)
        e_pred = en_predictor.predict(np.arange(doy[i] + 1, doy[i] + 1 + 365))
        duty_cycle = en_manager.calc_duty_cycle(
            doy[i], soc, e_pred)
        simulator.set_duty_cycle(duty_cycle)

    rup = enmanage.relative_underperformance(
        e_in[:len(doy)], budget, enmanage.profiles['uniform'](doy))

    efficiency = np.mean(budget)/np.mean(e_in[:len(doy)])

    assert(rup < 1.0)
    assert(efficiency > 0.0)


@pytest.fixture()
def battery():
    battery = enmanage.Battery(3500.0 * enmanage.CAP_FACTOR, 0.5)
    return battery


@pytest.fixture(params=[PREACT, LTENO, STEWMA])
def en_manager(fxt_dataset, battery, request):

    max_budget = max(fxt_dataset['data']['exposure']) * enmanage.PWR_FACTOR
    if(request.param is PREACT):
        en_manager = request.param(
            battery.capacity, enmanage.profiles['uniform']
        )
    elif(request.param is LTENO):
        en_manager = request.param(
            max_budget,
            battery.capacity,
            battery.get_eta_in(),
            battery.get_eta_out()
        )
    elif(request.param is STEWMA):
        en_manager = request.param(max_budget)

    return en_manager
