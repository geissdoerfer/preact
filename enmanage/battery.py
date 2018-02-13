import numpy as np


class BatteryException(Exception):
    pass


class Battery:
    def __init__(
            self, capacity, soc_init, model_parameters=None,
            estimation_errors=None):

        self.model_parameters = {
            'eta_in': 1.0,
            'eta_out': 1.0,
            'loss_rate': 0.0,
            'age_rate': 0.0
        }
        if model_parameters:
            self.model_parameters.update(model_parameters)

        self.estimation_errors = {
            'etas': 0.0,
            'age_rate': 0.0,
            'loss_rate': 0.0,
            'soc': (0.0, 0.0)
        }
        if estimation_errors:
            self.estimation_errors.update(estimation_errors)

        self.capacity = capacity
        self.soc = soc_init * self.capacity

        self.age_amount = capacity * self.model_parameters['age_rate']

    def charge(self, value):

        self.capacity = max(0.0, self.capacity - self.age_amount)

        loss = self.model_parameters['loss_rate'] * self.soc
        if(self.soc >= loss):
            e_wasted = self.soc
            self.soc = self.soc-loss
        else:
            e_wasted = self.soc
            self.soc = 0.0

        soc_prior = self.soc

        if value > 0:
            value_bat = value * self.model_parameters['eta_in']
            if self.soc + value_bat > self.capacity:
                self.soc = self.capacity
            else:
                self.soc = self.soc + value_bat

        else:
            value_bat = value / self.model_parameters['eta_out']
            if(self.soc + value_bat < 0):
                raise BatteryException("Battery undercharged")
            self.soc = self.soc + value_bat

        return self.soc - soc_prior

    def can_supply(self):
        return self.soc * self.model_parameters['eta_out']

    def can_absorb(self):
        return (self.capacity - self.soc) / self.model_parameters['eta_in']

    def get_eta_in(self):
        return (self.model_parameters['eta_in']
                * (1.0 + self.estimation_errors['etas']))

    def get_eta_out(self):
        return (self.model_parameters['eta_out']
                * (1.0 + self.estimation_errors['etas']))

    def get_loss_rate(self):
        return (self.model_parameters['loss_rate']
                * (1.0 + self.estimation_errors['loss_rate']))

    def get_age_rate(self):
        return (self.model_parameters['age_rate']
                * (1.0 + self.estimation_errors['age_rate']))

    def get_soc(self):
        return (self.soc
                * ((1.0 + (self.estimation_errors['soc'][0]
                   + np.random.randn()*self.estimation_errors['soc'][1])
                   / 100.0)))
