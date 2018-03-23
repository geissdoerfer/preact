import numpy as np
from scipy.optimize import minimize, curve_fit
import copy
import logging

from .prediction import EnergyPredictor
from .profiles import profiles

np.seterr(all='raise')


class EnergyManager(object):

    def calc_duty_cycle(self, *args):
        pass


class PredictiveManager(EnergyManager):
    def __init__(self, predictor):
        self.predictor = predictor

    def step(self, doy, soc):
        raise NotImplementedError()


class PIDController:
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

    def calculate(self, set_point, process_variable):

        e = (process_variable - set_point)
        self.e_sum += e
        e_d = e - self.e_prev
        self.e_prev = e

        output = (
            self.coefficients['k_p'] * e
            + self.coefficients['k_i'] * self.e_sum
            + self.coefficients['k_d'] * e_d
        )

        return output


class PREACT(PredictiveManager):
    def __init__(
            self, predictor, battery_capacity, battery_age_rate=0.0, **kwargs):

        PredictiveManager.__init__(self, predictor)

        self.controller = PIDController(
            kwargs.get('control_coefficients', None))

        if('utility_function' in kwargs.keys()):
            self.utility_function = kwargs['utility_function']
        else:
            def utility(doy):
                return np.ones(len(doy))
            self.utility_function = utility

        self.battery_capacity = battery_capacity
        self.battery_age_rate = battery_age_rate

        self.step_count = 0

        self.log = logging.getLogger("PREACT")

    def estimate_capacity(self, offset=0):
        return (
            self.battery_capacity
            * (1.0 - (self.step_count + offset) * self.battery_age_rate)
        )

    def calc_duty_cycle(self, doy, e_in, soc):

        self.predictor.update(doy, e_in)

        e_pred = self.predictor.predict(
            np.arange(doy + 1, doy + 1 + 365))

        e_req = self.utility_function(np.arange(doy + 1, doy + 1 + 365))
        f_req = np.mean(e_pred)/np.mean(e_req)

        d_soc_1y = np.cumsum(e_pred - f_req*e_req)

        p2p_1y = max(d_soc_1y)-min(d_soc_1y)
        if(p2p_1y < self.estimate_capacity() / 10000):
            return 1.0

        f_scale = min(1.0, self.estimate_capacity() / p2p_1y)

        offset = (self.estimate_capacity() - f_scale * p2p_1y) / 2

        self.soc_target = f_scale * (d_soc_1y[0] - min(d_soc_1y)) + offset

        self.step_count += 1

        duty_cycle = self.controller.calculate(
            self.soc_target / self.battery_capacity,
            (soc + e_pred[0]) / self.battery_capacity
        )

        return max(0.0, min(1.0, duty_cycle))


class PIDPM(EnergyManager):
    def __init__(
            self, battery_capacity,
            battery_age_rate, **kwargs):

        self.controller = PIDController(
            kwargs.get('control_coefficients', None))

        self.battery_capacity = battery_capacity
        self.battery_age_rate = battery_age_rate

        self.step_count = 0

        self.soc_target = 0.5

    def estimate_capacity(self, offset=0):
        return (
            self.battery_capacity
            * (1.0 - (self.step_count + offset) * self.battery_age_rate)
        )

    def calc_duty_cycle(self, doy, e_in, soc):

        self.step_count += 1

        duty_cycle = self.controller.calculate(
            self.soc_target,
            soc / self.estimate_capacity()
        )

        return max(0.0, min(1.0, duty_cycle))


class ENOMAX(EnergyManager):
    def __init__(
            self, battery_capacity,
            battery_age_rate, init_soc, **kwargs):

        self.battery_capacity = battery_capacity
        self.battery_age_rate = battery_age_rate

        self.params = np.array([2.0, -1.0, 1.0])
        self.duty_cycle = 0.5

        self.soc_target = 0.5

        self.learning_rate = kwargs.get('learning_rate', 0.5)
        self.alpha = kwargs.get('alpha', 1.0/24)
        self.beta = kwargs.get('beta', 0.25)

        self.phi = np.array([init_soc, self.duty_cycle, -self.soc_target])
        self.step_count = 0

    def estimate_capacity(self, offset=0):
        return (
            self.battery_capacity
            * (1.0 - (self.step_count + offset) * self.battery_age_rate)
        )

    def calc_duty_cycle(self, doy, e_in, soc):
        param_update = (
            self.learning_rate/np.dot(self.phi, self.phi) * self.phi
            * (soc/self.estimate_capacity() - np.dot(self.phi, self.params))
        )

        #print(self.phi, self.params, soc/self.estimate_capacity(), param_update)
        self.params += param_update
        duty_cycle = (
            (
                self.soc_target
                - self.params[0]*soc/self.estimate_capacity()
                + self.params[2]*self.soc_target
            )
            / self.params[1]
        )
        duty_cycle = max(0.0, min(1.0, duty_cycle))

        self.phi = np.array(
            [soc/self.estimate_capacity(), self.duty_cycle, self.soc_target]
        )

        self.duty_cycle += self.alpha * (duty_cycle - self.duty_cycle)

        rho = self.beta * duty_cycle + (1 - self.beta) * self.duty_cycle
        self.step_count += 1

        return rho


class STEWMA(PredictiveManager):
    def __init__(self, predictor, e_baseline, e_max_active, loss_rate=0.0):

        PredictiveManager.__init__(self, predictor)

        self.e_max_active = e_max_active
        self.e_baseline = e_baseline

        self.loss_rate = loss_rate

    def calc_duty_cycle(self, doy, e_in, soc):

        self.predictor.update(doy, e_in)
        e_pred = self.predictor.predict()

        dc = (
            (e_pred - self.loss_rate * soc - self.e_baseline)
            / self.e_max_active
        )
        return max(0.0, min(1.0, dc))


class LTENO(PredictiveManager):
    def __init__(
            self, predictor, e_baseline, e_max_active, battery_capacity,
            eta_bat_in=1.0, eta_bat_out=1.0):

        PredictiveManager.__init__(self, predictor)

        # Scale from nominal capacity
        self.battery_capacity = battery_capacity * eta_bat_out
        self.e_max_active = e_max_active
        self.e_baseline = e_baseline

        self.eta_bat_in = eta_bat_in
        self.d = predictor.d

        self.log = logging.getLogger("LT-ENO")
        self.log.debug(f'capacity: {self.battery_capacity:.{3}}')

    def calc_duty_cycle(self, doy, e_in, soc):

        self.predictor.update(doy, e_in)
        e_pred = self.predictor.predict(np.arange(365)) * self.eta_bat_in

        d = copy.copy(self.d)
        deficit = LTENO.deficit(e_pred, self.d)
        surplus = LTENO.surplus(e_pred, self.d)
        self.log.debug((
            f'initial: surplus={surplus:.{3}} deficit={deficit:.{3}}'
            f' alpha={self.predictor.alpha:.{3}}'
            f' {"under" if self.predictor.alpha>1.0 else "over"}estimated'
        ))
        while (deficit <= surplus) and (surplus < self.battery_capacity):
            d_ = copy.copy(d)

            if(self.predictor.alpha > 1):
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

        dc = (
            (e_pred[d[1] % 365] - self.e_baseline)
            / self.e_max_active
        )
        self.log.debug(f'd1={d[1]} dutycycle={dc:.{3}}')
        return max(0.0, min(1.0, dc))

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


class OptimalManager(EnergyManager):
    def __init__(self, t, e_ins, utility_function):

        self.controller = PIDController()

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
