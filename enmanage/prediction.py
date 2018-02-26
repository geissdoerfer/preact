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


class Model:
    def __init__(self, mfun, d_mfun):
        self.mfun = mfun
        self.d_mfun = d_mfun


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


class EnergyPredictor(object):

    def update(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class AST(EnergyPredictor):
    def __init__(self, **kwargs):

        def mfun_fixlat(lat):
            def mfun_a(d, a):
                return mfun(d, (a, lat))
            return mfun_a

        popt, copt = curve_fit(
            mfun_fixlat(kwargs['latitude']),
            kwargs['training_data']['doy'],
            kwargs['training_data']['e_in']
        )

        self.p = [popt[0], kwargs['latitude']]
        self.window_size = kwargs.get('window_size', 63)
        self.window = np.zeros(self.window_size)

        self.alpha = 1.0

        self.step_count = 0

        # Calculate intersections of energy model function and mean (See paper)
        e_ins_pred = self.predict(np.arange(365))
        self.d = np.empty(3, dtype=int)
        self.d[0] = np.argmax(
            np.diff(np.sign(e_ins_pred - np.mean(e_ins_pred))) > 0)
        self.d[2] = self.d[0] + 365
        search_region = np.arange(self.d[0], self.d[2], dtype=int) % 365
        sign_changes = np.diff(
            np.sign(
                e_ins_pred[search_region] - np.mean(e_ins_pred)
            )
        )
        self.d[1] = np.argmax(sign_changes < 0) + self.d[0]

    def update(self, d, y):

        # Calculate circular index for mini-batch buffer
        batch_ix = (self.step_count) % (self.window_size)
        # Put current value to buffer
        self.window[self.step_count % self.window_size] = y

        batch_size = min(self.window_size, self.step_count + 1)

        wndw_idx = np.arange(d + 1 - batch_size, d + 1)

        self.alpha = (
            np.mean(self.window[:batch_size])
            / np.mean(mfun(wndw_idx, self.p))
        )
        self.step_count += 1

    def predict(self, d):
        return self.alpha * mfun(d, self.p)


class CLAIRVOYANT(EnergyPredictor):

    def __init__(self, y_real, scale=1.0):

        self.y_real = y_real
        self.step_count = 0

    def update(self, x, y):
        self.step_count += 1

    def predict(self, x):
        if hasattr(x, "__len__"):
            return self.y_real[
                self.step_count + 1:self.step_count + 1 + len(x)]
        else:
            return self.y_real[self.step_count + 1]


class OPTMODEL(EnergyPredictor):

    def __init__(self, x_real, y_real, scale):

        self.scale = scale
        model = Model(mfun, (mfun_d_a, mfun_d_b))

        self.step_count = 0

        self.params = fit_optimal(
            x_real,
            y_real / self.scale,
            model
        )

    def predict(self, x):
        return mfun(x, self.params) * self.scale


class SGD(EnergyPredictor):
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

    def update(self, x, y):

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


class MBSGD(EnergyPredictor):
    @staticmethod
    def fn_eta(d):
        return 3.0/(1 + d/0.5)

    def __init__(self, scale=1.0, **kwargs):

        self.fn_eta = kwargs.get('fn_eta', MBSGD.fn_eta)
        self.batchsize = kwargs.get('batchsize', 1)
        self.momentum = kwargs.get('momentum', 0.375)
        self.params = kwargs.get('params_init', np.zeros(2))

        self.model = Model(mfun, (mfun_d_a, mfun_d_b))

        self.step_count = 0
        self.prev_update = np.zeros(2)
        self.batch_buffer = np.zeros(self.batchsize)

        self.scale = scale

    def update(self, x, y):

        y_scaled = y/self.scale

        eta = self.fn_eta(self.step_count)

        # Put current value to buffer
        self.batch_buffer[self.step_count % self.batchsize] = y_scaled

        batch_size = min(self.batchsize, self.step_count + 1)
        batch_idx = np.arange(x + 1 - batch_size, x + 1)

        # Partial derivatives
        dJs = np.zeros(len(self.params))

        # Iterate mini-batch
        for i in range(0, batch_size):
            # Get day by counting backwards
            d_i = (x + 1 - batch_size) + i
            # Prediction error
            e = self.model.mfun(d_i, self.params) - self.batch_buffer[i]
            for j in range(len(self.params)):
                dJs[j] += e * self.model.d_mfun[j](d_i, self.params)

        # Parameter update with momentum
        dW = - eta*dJs/(batch_size) + self.momentum*self.prev_update
        self.prev_update = dW
        self.params += dW

        self.step_count += 1

        # Make sure latitude is within [-pi;pi]
        if(self.params[1] > np.pi) or (self.params[1] < np.pi):
            self.params[1] = ((self.params[1] + np.pi) % (2*np.pi)) - np.pi

    def predict(self, x):
        return mfun(x, self.params)*self.scale


class EWMA(EnergyPredictor):
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 0.5)
        self.buffer = 0.0

    def update(self, x, y):
        self.buffer = self.alpha * y + (1.0 - self.alpha) * self.buffer

    def predict(self):
        return self.buffer
