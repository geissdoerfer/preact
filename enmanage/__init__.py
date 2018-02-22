import numpy as np
import logging

from .battery import Battery
from .profiles import profiles
from .managers import PREACT, LTENO, STEWMA
from .prediction import EWMA, MBSGD, AST, SGD, OPTMODEL, CLAIRVOYANT
from .constants import *


log = logging.getLogger("simulator")


class Consumer(object):
    def __init__(self, max_consumption):
        self.max_consumption = max_consumption

    def consume(self, duty_cycle):
        return duty_cycle * self.max_consumption


class Simulator(object):
    def __init__(
            self, consumer, battery, dc_init=0.5):

        self.battery = battery
        self.consumer = consumer

        self.duty_cycle = dc_init

    def set_duty_cycle(self, duty_cycle):
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))

    def step(self, n, e_in):

        e_out = self.consumer.consume(self.duty_cycle)
        e_net = e_in - e_out

        if(e_net < 0):
            e_charge_real = min(abs(e_net), self.battery.can_supply())
            self.battery.charge(-e_charge_real)
            e_in_real = e_in
            e_out_real = e_charge_real + e_in
        else:
            e_charge_real = min(e_net, self.battery.can_absorb())
            self.battery.charge(e_charge_real)
            e_in_real = e_charge_real + e_out
            e_out_real = e_out

        log.debug((
            f'e_in={e_in:.{3}} '
            f'e_in_real={e_in_real:.{3}} '
            f'e_out_real={e_in:.{3}} '
            f'soc={self.battery.soc/self.battery.capacity:.{3}}'
        ))

        return e_in_real, e_out_real, self.battery.get_soc()


def plan_capacity(doys, e_ins, latitude, eta_bat_in=1.0, eta_bat_out=1.0):

    astmodel = AST(doys, e_ins, latitude)
    e_pred = astmodel.predict(np.arange(365))

    surplus = LTENO.surplus(e_pred, astmodel.d)
    deficit = LTENO.deficit(e_pred, astmodel.d)

    log.debug(
        f'Planning capacity: surplus={surplus:.{3}}, deficit={deficit:.{3}}'
    )
    return (surplus + deficit) / 2


def relative_underperformance(e_in, e_out, utility):

    e_tmp = e_out - (utility / np.mean(utility) * np.mean(e_in))
    e_tmp[e_tmp >= 0.0] = 0.0
    return - np.mean(e_tmp) / np.mean(e_in)
