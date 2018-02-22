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
            self, manager, predictor, consumer, battery, dc_init=0.5):

        self.battery = battery
        self.consumer = consumer
        self.predictor = predictor
        self.manager = manager

        self.next_duty_cycle = dc_init

    def simulate_consumption(self, e_in, duty_cycle):
        e_out = self.consumer.consume(duty_cycle)
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

        duty_cycle_real = (
            (e_out_real - self.consumer.consume(0.0))
            / self.consumer.consume(1.0)
        )
        return e_in_real, e_out_real, duty_cycle_real

    def simulate_budgeting(self, doy, e_in_real):
        self.predictor.update(doy, e_in_real)
        duty_cycle = min(
            1.0, max(0.0, self.manager.step(doy, self.battery.get_soc()))
        )
        return duty_cycle

    def step(self, doy, e_in):

        e_in_real, e_out_real, duty_cycle_real = self.simulate_consumption(
            e_in, self.next_duty_cycle)

        self.next_duty_cycle = self.simulate_budgeting(doy, e_in_real)

        log.debug((
            f'e_in={e_in:.{3}} '
            f'e_in_real={e_in_real:.{3}} '
            f'e_out_real={e_in:.{3}} '
            f'soc={self.battery.soc/self.battery.capacity:.{3}}'
        ))

        return self.battery.soc, duty_cycle_real, e_in_real, e_out_real

    def run(self, doys, e_ins):
        budget = np.zeros(len(doys))
        soc = np.zeros(len(doys))
        duty_cycle = np.zeros(len(doys))
        for i, (doy, e_in) in enumerate(zip(doys, e_ins)):
            soc[i], duty_cycle[i], e_in_real, budget[i] = self.step(
                doy, e_in)

        return soc, budget, duty_cycle


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
