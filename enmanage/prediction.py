import numpy as np
from scipy.optimize import minimize, curve_fit


def radiance(d):
    return 1367.0 * (1 + 0.033 * np.cos(2 * np.pi * d / 365.25))


def sunrise_angle(d, lat):
    arg = -np.tan(lat)*np.tan(solar_declination(d))

    if hasattr(arg, "__len__"):
        arg[arg > 1.0] = 1.0
        arg[arg < -1.0] = -1.0
    else:
        arg = max(-1.0, min(1.0, arg))

    return np.arccos(arg)


def solar_declination(d):
    return 23.45 / 180.0 * np.pi * np.sin(2 * np.pi * (284.0 + d) / 365.0)


def mfun_unzip(d, a, b):
    return mfun(d, (a, b))


def mfun(d, params):
    return (
        params[0]*radiance(d)*(1.0/(300*np.pi)) *
        (
            np.sin(params[1])
            * np.sin(solar_declination(d))
            * sunrise_angle(d, params[1])
            + np.cos(params[1])
            * np.cos(solar_declination(d))
            * np.sin(sunrise_angle(d, params[1]))
        )
            )


def mfun_d_a(d, params):
    return (
        radiance(d)*(1.0/(300*np.pi)) *
        (
            np.sin(params[1])
            * np.sin(solar_declination(d))
            * sunrise_angle(d, params[1])
            + np.cos(params[1])
            * np.cos(solar_declination(d))
            * np.sin(sunrise_angle(d, params[1]))
        )
            )


def mfun_d_b(d, params):
    return (
        params[0] * radiance(d) * (1.0 / (300 * np.pi)) *
        (
            np.cos(params[1])
            * np.sin(solar_declination(d))
            * sunrise_angle(d, params[1])
            - np.sin(params[1])
            * np.cos(solar_declination(d))
            * np.sin(sunrise_angle(d, params[1]))
        )
            )


def fit_optimal(ds, ys, model):

    def constr_lat(params):
        return np.pi-abs(params[1])

    def constr_a(params):
        return params[0]

    def obj(params, ds, ys, model):
        y_est = model.mfun(ds, params)
        return np.mean((y_est-ys)**2)

    x0 = [1.0, 1.0]
    constraints = [
        {
            'type': 'ineq',
            'fun': constr_lat
        },
        {
            'type': 'ineq',
            'fun': constr_a
        }]

    res = minimize(
        obj, x0, args=(ds, ys, model), method='SLSQP', constraints=constraints)
    return res.x


class AST(object):
    def __init__(self, t_training, e_in_training, latitude, window_size=63):

        def mfun_fixlat(lat):
            def mfun_a(d, a):
                return mfun(d, (a, lat))
            return mfun_a
        popt, copt = curve_fit(
            mfun_fixlat(latitude), t_training, e_in_training)

        self.p = [popt[0], latitude]
        self.window = np.zeros(window_size)
        self.wndw_size = window_size
        self.alpha = 1.0

        self.step_count = 0

    def step(self, d, y):

        # Calculate circular index for mini-batch buffer
        batch_ix = (self.step_count + 1) % (self.wndw_size)
        # Put current value to buffer
        self.window[batch_ix-1] = y

        real_batch_size = (
            self.step_count + 1
            - max(0, (self.step_count + 1) - self.wndw_size)
        )

        wndw_start = max(0, d+1-real_batch_size)
        wndw_idx = np.arange(wndw_start, d+1)

        self.alpha = (
            np.mean(self.window[0:real_batch_size])
            / np.mean(mfun(wndw_idx, self.p))
        )
        self.step_count += 1

    def predict(self, d):
        return self.alpha * mfun(d, self.p)


class OPTMODEL(object):

    def __init__(self, y_real, scale=1.0):

        self.scale = scale
        self.model = Model(mfun, (mfun_d_a, mfun_d_b))
        self.y_real = y_real
        self.step_count = 0

        self.params = np.zeros(2)

    def step(self, x, y):

        d_1y = np.arange(x+1, x+1+365)
        ix_1y = np.arange(self.step_count + 1, self.step_count + 1 + 365)

        self.params = fit_optimal(
            d_1y,
            self.y_real[ix_1y] / self.scale,
            self.model
        )

        self.step_count += 1

    def predict(self, x):
        return mfun(x, self.params) * self.scale


class SGD(object):
    @staticmethod
    def fn_eta(d):
        return 1.5/(d+0.5)

    def __init__(self, params_init=None, fn_eta=None, scale=1.0):

        if fn_eta is None:
            self.fn_eta = MBSGD.fn_eta
        else:
            self.fn_eta = fn_eta

        if params_init is None:
            self.params = np.array([2.0, 0.0])
        else:
            self.params = params_init

        self.scale = scale

        self.model = Model(mfun, (mfun_d_a, mfun_d_b))

        self.step_count = 0

    def step(self, x, y):

        y_scaled = y/self.scale

        eta = self.fn_eta(self.step_count)

        # Partial derivatives
        dJs = np.zeros(len(self.params))

        e = self.model.mfun(x, self.params) - y_scaled
        for j in range(len(self.params)):
            dJs[j] = e * self.model.d_mfun[j](x, self.params)

        self.params -= eta*dJs

        self.step_count += 1

        # Make sure latitude is within [-pi;pi]
        self.params[1] = (self.params[1] + np.pi) % (2*np.pi) - np.pi

    def predict(self, x):
        return mfun(x, self.params) * self.scale


class MBSGD(object):
    @staticmethod
    def fn_eta(d):
        return 1.5/(d+0.5)

    def __init__(
            self, scale=1.0, fn_eta=None, batchsize=1, momentum=0.375,
            params_init=None):

        if fn_eta is None:
            self.fn_eta = MBSGD.fn_eta
        else:
            self.fn_eta = fn_eta

        self.momentum = momentum
        self.batchsize = batchsize

        if params_init is None:
            self.params = np.zeros(2)
        else:
            self.params = params_init

        self.model = Model(mfun, (mfun_d_a, mfun_d_b))

        self.step_count = 0
        self.prev_update = np.zeros(2)
        self.batch_buffer = np.zeros(batchsize)

        self.scale = scale

    def step(self, x, y):

        y_scaled = y/self.scale

        eta = self.fn_eta(self.step_count)

        # Calculate circular index for mini-batch buffer
        batch_ix = (self.step_count + 1) % (self.batchsize)
        # Put current value to buffer
        self.batch_buffer[batch_ix-1] = y_scaled
        real_batch_size = (
            self.step_count + 1
            - max(0, self.step_count + 1 - self.batchsize)
        )

        # Partial derivatives
        dJs = np.zeros(len(self.params))

        # Iterate mini-batch
        for i in range(batch_ix-real_batch_size, batch_ix):
            # Get day by counting backwards
            d_i = x-(batch_ix-1-i)
            # Prediction error
            e = self.model.mfun(d_i, self.params) - self.batch_buffer[i]
            for j in range(len(self.params)):
                dJs[j] += e * self.model.d_mfun[j](d_i, self.params)

        # Parameter update with momentum
        dW = - eta*dJs/(real_batch_size) + self.momentum*self.prev_update
        self.prev_update = dW
        self.params += dW

        self.step_count += 1

        # Make sure latitude is within [-pi;pi]
        p_tmp = self.params[1] + np.pi
        if(p_tmp > 2*np.pi) or (p_tmp < 0.0):
            p_tmp = p_tmp % (2*np.pi)

        self.params[1] = p_tmp - np.pi

    def predict(self, x):
        return mfun(x, self.params)*self.scale


class EWMA(object):
    def __init__(self, alpha=0.5):
        self.alpha = 0.5
        self.buffer = 0.0

    def step(self, x, y):
        self.buffer = self.alpha * y + (1.0 - self.alpha) * self.buffer

    def predict(self, x):
        return self.buffer


class Model:
    def __init__(self, mfun, d_mfun):
        self.mfun = mfun
        self.d_mfun = d_mfun
