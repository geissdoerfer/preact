import numpy as np
import pytest
import enmanage
from enmanage.managers import *
from nasadata import fxt_dataset


def test_manage(fxt_dataset, battery, en_manager):

    doy = np.array(fxt_dataset['data'].index.dayofyear)[:365]
    e_in = fxt_dataset['data']['exposure'].as_matrix()

    en_predictor = enmanage.CLAIRVOYANT(e_in)

    simulator = enmanage.Simulator(en_manager, en_predictor, battery)

    budget = np.zeros(len(doy))
    for i in range(len(doy)):
        budget[i] = simulator.step(doy[i], e_in[i])

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

    if(request.param is PREACT):
        en_manager = request.param(
            battery.capacity, enmanage.profiles['uniform']
        )
    elif(request.param is LTENO):
        en_manager = request.param(
            battery.capacity, battery.get_eta_in(), battery.get_eta_out()
        )
    else:
        en_manager = request.param()

    return en_manager
