from pathlib import Path
import parse
import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import multiprocessing
import matplotlib.pyplot as plt
import logging

import enmanage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kgain")

BASE_PATH = Path(__file__).resolve().parent.parent

CAPACITY = 4522.7 * enmanage.CAP_FACTOR
PWR_FACTOR = enmanage.PWR_FACTOR

TRAINING_OFFSET = 120

INIT_SOC = 0.5


def utility(doys):
    return np.ones(len(doys))


class MyConsumer(enmanage.Consumer):
    def __init__(self, max_consumption):
        self.max_consumption = max_consumption

    def step(self, duty_cycle):
        return self.max_consumption * duty_cycle


def simulate(
        control_coefficients, doys, e_ins, res_q):

    predictor = enmanage.MBSGD(scale=PWR_FACTOR)

    for i in range(TRAINING_OFFSET):
        predictor.step(doys[i], e_ins[i])
        e_ins_pred = predictor.predict(doys)

    battery = enmanage.Battery(CAPACITY, INIT_SOC)
    manager = enmanage.PREACT(
        battery.capacity, utility, control_coefficients)

    consumer = MyConsumer(max(e_ins))
    simulator = enmanage.Simulator(consumer, battery)

    budget = np.zeros(len(doys) - TRAINING_OFFSET)

    for i in range(TRAINING_OFFSET, len(doys)):
        e_in_real, budget[i-TRAINING_OFFSET], soc = simulator.step(
            doys[i], e_ins[i])
        predictor.step(doys[i], e_in_real)
        e_pred = predictor.predict(np.arange(doys[i] + 1, doys[i] + 1 + 365))
        duty_cycle = manager.calc_duty_cycle(doys[i], soc, e_pred)
        simulator.set_duty_cycle(duty_cycle)

    rup = enmanage.relative_underperformance(
        e_ins[TRAINING_OFFSET:], budget, utility(doys[TRAINING_OFFSET:]))

    res_q.put(rup)


def run(ks):
    control_coefficients = {'k_p': ks[0], 'k_i': ks[1], 'k_d': ks[2]}
    rups = list()

    res_q = multiprocessing.Queue()
    jobs = list()

    for location_code in ['A_MRA', 'B_ASP', 'C_BLN']:
        dataset = get_dataset(location_code)
        e_ins = dataset['data']['exposure'].as_matrix()[:3*365] * PWR_FACTOR
        doys = np.array(dataset['data'].index.dayofyear)[:3*365]

        p = multiprocessing.Process(
            target=simulate,
            args=(control_coefficients, doys, e_ins, res_q)
        )

        p.start()
        jobs.append(p)

    for p in jobs:
        p.join()

    res_l = list()
    while not res_q.empty():
        res_l.append(res_q.get_nowait())

    log.info(
        (f'k_p={ks[0]:.{3}}, k_i={ks[1]:.{3}}, k_d={ks[2]:.{3}}'
         f' UP: {100*np.mean(res_l):.{3}}')
    )
    return np.mean(res_l)


def nm_optimize(maxiter):

    simplex_init = np.array(
        [[0.0, 0.0, 0.0], [.1, 0.0, 0.0], [0.0, .1, 0.0],
         [0.0, 0.0, .1]]
    )
    res = minimize(
            run, np.zeros(3), method='Nelder-Mead',
            options={
                'maxiter': maxiter, 'fatol': 0.0001,
                'initial_simplex': simplex_init})

    ret_dict = {
        'k_p': res.x[0], 'k_i': res.x[1], 'k_d': res.x[2], 'rup': res.fun}
    log.info(str(ret_dict))
    return ret_dict


def get_dataset(location_code):
    filepath = (
        BASE_PATH
        / 'data'
        / 'energy'
        / f'{ location_code }.txt'
    )

    df = pd.read_csv(
        filepath, delim_whitespace=True, skiprows=8, parse_dates=[[0, 1, 2]],
        date_parser=lambda *columns: datetime(*map(int, columns)))

    df.columns = ['date', 'exposure']
    df.set_index(df['date'], inplace=True)
    df.drop('date', 1, inplace=True)

    with open(filepath, 'r') as f:
        for _ in range(3):
            line = f.readline()
        latitude = parse.search('Latitude {:f}', line)[0] / 360.0 * 2 * np.pi

    return {'data': df, 'latitude': latitude}


if __name__ == '__main__':

    nm_optimize(1000)
