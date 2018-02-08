import numpy as np

from .battery import Battery
from .profiles import profiles
from .managers import PREACT, LTENO, EWMA
from .prediction import Model, fit_optimal, MBSGD, AST, SGD


class Simulator(object):
    def __init__(energy_manager, battery):
        self.energy_manager = energy_manager
        self.battery = battery

        self.e_out_prev = 0.0

    def step(n, e_in):
        e_net = e_in-self.e_out_prev

        if(e_net < 0):
            e_charge_real = min(abs(e_net), self.battery.can_supply())
            self.battery.charge(-e_charge_real)
            e_in_real = e_in
            e_out_real = e_charge_real + e_in
        else:
            e_in_real = self.battery.charge(e_net)
            e_out_real = self.e_out_prev

        self.e_out_prev = max(
            0.0, self.energy_manager.calc_budget(
                n, e_in_real, self.battery.get_soc()))

        return e_out_real, self.battery.soc


def get_error(e_out, e_req, e_in):
    b = np.mean(e_in)/np.mean(e_req)
    e_tmp = e_out-b*e_req
    e_tmp[e_tmp >= 0.0] = 0.0
    return -np.mean(e_tmp)/np.mean(e_in)
