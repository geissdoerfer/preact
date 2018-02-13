import numpy as np
import logging

from .battery import Battery
from .profiles import profiles
from .managers import PREACT, LTENO, STEWMA
from .prediction import EWMA, MBSGD, AST, SGD, OPTMODEL, CLAIRVOYANT
from .constants import *


log = logging.getLogger("simulator")


class Simulator(object):
    def __init__(self, energy_manager, energy_predictor, battery):
        self.energy_manager = energy_manager
        self.battery = battery
        self.energy_predictor = energy_predictor

        self.e_out_prev = 0.0

    def step(self, n, e_in):
        e_net = e_in - self.e_out_prev

        if(e_net < 0):
            e_charge_real = min(abs(e_net), self.battery.can_supply())
            self.battery.charge(-e_charge_real)
            e_in_real = e_in
            e_out_real = e_charge_real + e_in
        else:
            e_charge_real = min(e_net, self.battery.can_absorb())
            self.battery.charge(e_charge_real)
            e_in_real = e_charge_real + self.e_out_prev
            e_out_real = self.e_out_prev

        self.energy_predictor.step(n, e_in_real)
        e_pred = self.energy_predictor.predict(np.arange(n + 1, n + 1 + 365))

        budget = self.energy_manager.calc_budget(
            n, self.battery.get_soc(), e_pred)

        self.e_out_prev = max(0.0, budget)
        log.debug((
            f'e_in={e_in:.{3}} '
            f'e_in_real={e_in_real:.{3}} '
            f'e_out_real={e_in:.{3}} '
            f'soc={self.battery.soc/self.battery.capacity:.{3}}'
        ))
        return e_out_real


def plan_capacity(eta_bat_in, eta_bat_out, e_pred):

    budget = np.mean(e_pred)
    e_d = e_pred - budget
    e_d[e_d > 0] *= eta_bat_in
    e_d[e_d < 0] /= eta_bat_out
    soc_delta = np.cumsum(e_d)
    return max(soc_delta) - min(soc_delta)


def relative_underperformance(e_in, e_out, utility):

    e_tmp = e_out - (utility / np.mean(utility) * np.mean(e_in))
    e_tmp[e_tmp >= 0.0] = 0.0
    return - np.mean(e_tmp) / np.mean(e_in)
