import numpy as np
from scipy.optimize import minimize, curve_fit
import copy
import logging

from .prediction import MBSGD, AST
from .profiles import profiles
from . import constants as C


class EnergyManager:
    pass


class ControlledManager:
    def __init__(self, coefficients=None):
        self.e_sum = 0.0
        self.e_prev = 0.0

        self.coefficients = {
            'k_p': 1.0,
            'k_i': 0.0,
            'k_d': 0.0
        }
        if coefficients:
            self.coefficients.update(coefficients)

    def soc_control(self, capacity, soc_target, soc):

        e = (soc - soc_target) / capacity
        self.e_sum += e
        e_d = e - self.e_prev
        self.e_prev = e

        dc = (
            self.coefficients['k_p'] * e
            + self.coefficients['k_i'] * self.e_sum
            + self.coefficients['k_d'] * e_d
        )

        return dc


class PREACT(EnergyManager, ControlledManager):
    def __init__(
            self, battery_capacity, utility_function,
            control_coefficients=None, battery_age_rate=0.0):

        ControlledManager.__init__(self, control_coefficients)

        self.utility_function = utility_function
        self.battery_capacity = battery_capacity
        self.battery_age_rate = battery_age_rate

        self.step_count = 0

        self.log = logging.getLogger("PREACT")

    def estimate_capacity(self, offset=0):
        return (
            self.battery_capacity
            * (1.0 - (self.step_count + offset) * self.battery_age_rate)
        )

    def calc_duty_cycle(self, n, soc, e_pred):

        d_1y = np.arange(n + 1, n + 1 + 365)

        e_req = self.utility_function(d_1y)
        f_req = np.mean(e_pred)/np.mean(e_req)
        e_d_1y = e_pred - f_req*e_req

        d_soc_1y = np.cumsum(e_d_1y)
        p2p_1y = max(d_soc_1y)-min(d_soc_1y)

        min_capacity_1y = min(self.estimate_capacity(np.arange(365)))

        f_scale = min_capacity_1y / p2p_1y

        if(f_scale < 1.0):
            d_soc_1y = f_scale * d_soc_1y
            offset = (self.estimate_capacity() - f_scale * p2p_1y) / 2

        else:
            offset = (self.estimate_capacity() - p2p_1y) / 2

        self.soc_target = d_soc_1y[0] + offset - min(d_soc_1y)

        self.step_count += 1

        duty_cycle = self.soc_control(
            self.estimate_capacity(), self.soc_target, soc + e_pred[0])

        return max(0.0, min(1.0, duty_cycle))


class STEWMA(EnergyManager):
    def __init__(self, e_out_max):
        self.e_out_max = e_out_max

    def calc_duty_cycle(self, e_pred):
        return e_pred / self.e_out_max


class LTENO(EnergyManager):
    def __init__(
            self, e_out_max, battery_capacity, d,
            eta_bat_in=1.0, eta_bat_out=1.0):

        # Scale from nominal capacity
        self.battery_capacity = battery_capacity * eta_bat_out
        self.e_out_max = e_out_max
        self.eta_bat_in = eta_bat_in
        self.d = d

        self.log = logging.getLogger("LT-ENO")
        self.log.debug(f'capacity: {self.battery_capacity:.{3}}')

    def calc_duty_cycle(self, e_pred, alpha):

        e_pred = e_pred * self.eta_bat_in
        d = copy.copy(self.d)
        deficit = LTENO.deficit(e_pred, self.d)
        surplus = LTENO.surplus(e_pred, self.d)
        self.log.debug((
            f'initial: surplus={surplus:.{3}} deficit={deficit:.{3}}'
            f' alpha={alpha:.{3}} {"under" if alpha>1.0 else "over"}estimated'
        ))
        while (deficit <= surplus) and (surplus < self.battery_capacity):
            d_ = copy.copy(d)

            if(alpha > 1):
                self.log.debug('decreasing surplus')
                d[0] = d[0] + 1
                d[1] = d[1] - 1
                d[2] = d[2] + 1
            else:
                self.log.debug('decreasing deficit')
                d[0] = d[0] - 1
                d[1] = d[1] + 1
                d[2] = d[2] - 1

            deficit = LTENO.deficit(e_pred, d)
            surplus = LTENO.surplus(e_pred, d)
            if(surplus < deficit):
                self.log.debug('stopping: surplus < deficit')
                d = copy.copy(d_)
                surplus = LTENO.surplus(e_pred, d)
                break

        duty_cycle = e_pred[d[1] % 365] / self.e_out_max
        self.log.debug(f'd1={d[1]} dutycycle={duty_cycle:.{3}}')
        return duty_cycle

    @staticmethod
    def deficit(e_pred, d):
        deficit_region = np.arange(d[1], d[2]) % 365
        deficit = (
            e_pred[d[1] % 365] * (d[2] - d[1] + 1)
            - np.sum(e_pred[deficit_region])
        )
        return deficit

    @staticmethod
    def surplus(e_pred, d):
        surplus_region = np.arange(d[0], d[1]) % 365
        surplus = (
            np.sum(e_pred[surplus_region])
            - e_pred[d[0]] * (d[1] - d[0] + 1)
        )
        return surplus


class OptimalManager(EnergyManager, ControlledManager):
    def __init__(self, t, e_ins, utility_function):
        ControlledManager.__init__(self)

        def get_soc_delta(p, e_in):
            return np.cumsum(e_in - p)

        def constr_amp(p, e_in, c_battery):
            soc_delta = get_soc_delta(p, e_in)
            return c_battery-(max(soc_delta)-min(soc_delta))

        def constr_eno(p, e_in):
            delta = get_soc_delta(p, e_in)
            return delta[-1]-delta[0]

        def objective(p, e_ideal):
            err = p-e_ideal
            uc = err[err <= 0]
            if(len(uc) == 0):
                return 0.0
            return np.mean(uc**2)/np.mean(e_ideal)*100.0

        e_req = mreq(t)
        e_in = e_ins[t]

        f_req = np.mean(e_in)/np.mean(e_req)
        e_ideal = f_req * e_req

        soc_tmp = np.cumsum(e_in-e_ideal)
        f_scale = capacity/((max(soc_tmp)-min(soc_tmp)))
        if(f_scale < 1.0):
            soc_tmp = soc_tmp * f_scale

        x0 = e_in-np.append([0.0], np.diff(soc_tmp))
        constraints = [
            {
                'type': 'ineq',
                'fun': constr_amp,
                'args': (e_in, capacity)
            }
        ]
        # {'type':'ineq','fun':constr_eno,'args': [e_in]}]
        bounds = [(0.0, None) for _ in x0]
        res = minimize(
            objective, x0, args=(e_ideal), bounds=bounds, method='SLSQP',
            tol=1e-05, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-05})
        # print res.nit, res.message
        # print objective(res.x, e_ideal)

        soc_delta = get_soc_delta(res.x, e_in)
        soc_offset = -min(soc_delta) + constr_amp(res.x, e_in, capacity)/2
        target_soc = soc_delta + soc_offset

        self.e_out_prev = res.x[0]
        self.battery.soc = target_soc[0]

        # Und moechte in Wirklichkeit hierhin: soc_delta[1]+ideal_init
        # ideal_init = constr_amp(res.x, e_in, capacity)/2 - min(soc_delta)
        # self.battery.soc = ideal_init

        self.budget = np.append(res.x[1:], 0.0)

    def calc_duty_cycle(self, n, e_in_real, soc):

        # print self.pred_soc[n], self.battery.soc
        # dsoc = self.soc_control(self.target_soc[n-self.t_offset+1], soc)
        # return self.e_pred[n-self.t_offset]-dsoc
        return self.budget[n]
