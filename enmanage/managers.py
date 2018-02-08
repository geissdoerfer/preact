import numpy as np
from scipy.optimize import minimize, curve_fit

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

        e = (soc_target - soc) / capacity
        self.e_sum += e
        e_d = e - self.e_prev
        self.e_prev = e

        dsoc = (
            self.coefficients['k_p'] * e
            + self.coefficients['k_i'] * self.e_sum
            + self.coefficients['k_d'] * e_d
        )

        return dsoc * capacity


class PREACT(EnergyManager, ControlledManager):
    def __init__(
            self, control_coefficients, utility_function, power_factor=1.0):
        ControlledManager.__init__(self, control_coefficients)

        self.utility_function = utility_function
        self.en_predictor = MBSGD(scale=power_factor)

    def calc_budget(self, n, e_in_real, soc):

        self.en_predictor.step(n, e_in_real)

        d_1y = np.arange(n+1, n+1+365)

        e_in_pred = self.en_predictor.predict(d_1y)

        e_req = self.utility_function(d_1y)

        f_req = np.mean(e_in_pred)/np.mean(e_req)

        e_d_1y = e_in_pred - f_req*e_req

        d_soc_1y = np.cumsum(e_d_1y)

        p2p_1y = max(d_soc_1y)-min(d_soc_1y)

        # The 0.9 lets the algorithm operate a bit more conservative
        f_scale = 0.9*min(self.get_capacity(d_1y))/p2p_1y

        if(f_scale < 1.0):
            d_soc_1y = f_scale * d_soc_1y
            offset = (self.capacity - f_scale*p2p_1y)/2

        else:
            offset = (self.capacity - p2p_1y)/2

        soc_target = d_soc_1y[0] + offset - min(d_soc_1y)

        dsoc = self.soc_control(soc_target, soc)
        return e_in_pred[0]-dsoc


class EWMA(EnergyManager):
    def __init__(alpha=0.5):

        self.buf = 0.0
        self.alpha = alpha

    def calc_budget(self, n, e_in_real, soc):

        self.buf = self.alpha * e_in_real + (1.0 - self.alpha) * self.buf

        return self.buf


class LTENO(EnergyManager):
    def __init__(
            training_data, latitude, power_factor=C.PWR_FACTOR,
            window_size=63):

        self.power_factor = power_factor
        self.en_predictor = AST(
            training_data['t'], training_data['e_in']/self.power_factor,
            latitude, window_size)

    @staticmethod
    def constr_surplus(e_out, e_pred_1y, capacity, eta_bat_in):
        e_d = e_pred_1y - e_out
        e_d[e_d > 0] *= eta_bat_in
        surp = np.sum(e_d[e_d > 0])
        return surp - capacity

    @staticmethod
    def constr_deficit(e_out, e_pred_1y, capacity, eta_bat_out):
        e_d = e_pred_1y - e_out
        e_d[e_d < 0] /= eta_bat_out
        defi = -np.sum(e_d[e_d < 0])
        return capacity - defi

    @staticmethod
    def obj_eno(e_out, e_pred_1y):
        return abs(np.mean(e_pred_1y)-e_out)

    def plan_capacity(self):
        ds_1y = np.arange(365)
        e_in = self.en_predictor.predict(ds_1y)*self.power_factor
        budget = np.mean(e_in)
        e_d = e_in-budget
        e_d[e_d > 0] *= self.eta_bat_in
        e_d[e_d < 0] /= self.eta_bat_out
        soc_delta = np.cumsum(e_d)
        return max(soc_delta) - min(soc_delta)

    def calc_budget(self, n, e_in_real, soc):

        self.en_predictor.step(n, e_in_real/self.power_factor)
        d_1y = np.arange(0, 365)

        e_pred_1y = self.en_predictor.predict(d_1y) * self.power_factor

        x0 = np.mean(e_pred_1y)
        constraints = [
            {
                'type': 'ineq',
                'fun': ETHManager.constr_surplus,
                'args': [e_pred_1y, self.capacity, self.eta_bat_in]
            },
            {
                'type': 'ineq',
                'fun': ETHManager.constr_deficit,
                'args': [e_pred_1y, self.capacity, self.eta_bat_out]
            }]
        res = minimize(
            LTENO.obj_eno,
            x0,
            args=(e_pred_1y),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 250, 'ftol': 1e-04}
            )

        return res.x[0]


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

    def calc_budget(self, n, e_in_real, soc):

        # print self.pred_soc[n], self.battery.soc
        # dsoc = self.soc_control(self.target_soc[n-self.t_offset+1], soc)
        # return self.e_pred[n-self.t_offset]-dsoc
        return self.budget[n]
