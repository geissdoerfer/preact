import numpy as np
import logging
import yaml
import collections
import copy
from pkg_resources import Requirement, resource_filename

from .battery import Battery
from .profiles import profiles
from .managers import PREACT, LTENO, STEWMA, PIDPM
from .prediction import EWMA, MBSGD, AST, SGD, OPTMODEL, CLAIRVOYANT

base_cfg_path = resource_filename(__name__, "simulator_config.yml")
log = logging.getLogger("simulator")


def dict_merge(dct, merge_dct):
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class Consumer(object):
    def __init__(self, e_max_active, e_baseline=0.0):
        self.e_max_active = e_max_active
        self.e_baseline = e_baseline

    def consume(self, duty_cycle):
        return self.e_baseline + duty_cycle * self.e_max_active


class Simulator(object):
    def __init__(
            self, manager, consumer, battery, dc_init=0.5,
            pwr_factor=1.0):

        self.battery = battery
        self.consumer = consumer
        self.manager = manager

        self.next_duty_cycle = dc_init
        self.pwr_factor = pwr_factor

    @staticmethod
    def get_config(config_path=None, config_dict={}):

        with open(base_cfg_path, 'r') as stream:
            base_config = yaml.load(stream)

        if config_path is not None:
            with open(config_path, 'r') as stream:
                config = yaml.load(stream)

            dict_merge(base_config, config)

        dict_merge(base_config, config_dict)
        return base_config

    @classmethod
    def calc_pwr_factor(cls, config_path=None, config_dict={}):
        config = cls.get_config(config_path, config_dict)

        pwr_factor = (
            config['harvesting']['a_panel']
            * config['harvesting']['eta_vc_in']
            * config['harvesting']['eta_panel']
            * 1000.0
        )
        return pwr_factor

    @classmethod
    def plan_capacity(
            cls, doys, e_ins, latitude, config_path=None, config_dict={}):
        config = cls.get_config(config_path, config_dict)

        pwr_factor = cls.calc_pwr_factor(config_dict=config)

        astmodel = AST(
            training_data={'doy': doys, 'e_in': e_ins * pwr_factor},
            latitude=latitude
        )
        e_pred = astmodel.predict(np.arange(365))

        surplus = LTENO.surplus(e_pred, astmodel.d)
        deficit = LTENO.deficit(e_pred, astmodel.d)

        capacity_mah = (
            (surplus + deficit) / 2
            / config['harvesting']['voltage']
            * 1000
        ) * config['battery']['model_parameters']['eta_out']

        return capacity_mah

    @classmethod
    def from_config(
            cls, manager_cls, manager_args=None,
            predictor_args=None, config_path=None, config_dict={},
            consumer=None):

        config = cls.get_config(config_path, config_dict)

        pwr_factor = cls.calc_pwr_factor(config_dict=config)

        capacity_wh = (
            config['battery']['capacity_mah']
            * config['harvesting']['voltage']
            / 1000
        )

        battery = Battery(
            capacity_wh,
            config['battery']['soc_init'],
            config['battery']['model_parameters'],
            config['battery']['estimation_errors']
        )

        if consumer is None:
            consumer = Consumer(
                config['consumer']['e_max_active'],
                config['consumer']['e_baseline']
            )

        if(manager_cls is LTENO):
            predictor_args = copy.deepcopy(predictor_args)
            predictor_args['training_data']['e_in'] *= pwr_factor
            predictor = AST(**predictor_args)

            manager = LTENO(
                predictor,
                config['consumer']['e_baseline'],
                config['consumer']['e_max_active'],
                battery.capacity,
                battery.get_eta_in(),
                battery.get_eta_out()
            )
        elif(manager_cls == PREACT):
            predictor_args = copy.deepcopy(predictor_args)
            predictor_args['training_data']['e_in'] *= pwr_factor
            predictor = AST(**predictor_args)

            manager = PREACT(
                predictor,
                battery.capacity,
                battery.get_age_rate(),
                **manager_args
            )

        elif(manager_cls == STEWMA):
            predictor = EWMA()

            manager = STEWMA(
                predictor,
                config['consumer']['e_baseline'],
                config['consumer']['e_max_active'],
                battery.get_loss_rate()
            )
        elif(manager_cls == PIDPM):
            predictor = None

            manager = PIDPM(
                battery.capacity,
                battery.get_age_rate(),
                **manager_args
            )
        else:
            manager = manager_cls(manager_args)

        return cls(
            manager, consumer, battery,
            config['simulator']['dc_init'],
            pwr_factor
        )

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
            / (self.consumer.consume(1.0) - self.consumer.consume(0.0))
        )

        return e_in_real, e_out_real, max(0.0, min(1.0, duty_cycle_real))

    def step(self, doy, e_in):

        e_in_real, e_out_real, duty_cycle_real = self.simulate_consumption(
            e_in * self.pwr_factor, self.next_duty_cycle)

        self.next_duty_cycle = max(0.0, min(
            1.0,
            self.manager.calc_duty_cycle(
                doy, e_in_real, self.battery.get_soc()
            )
        ))

        log.debug((
            f'e_in={e_in:.3} '
            f'e_in_real={e_in_real:.3} '
            f'e_out_real={e_out_real:.3} '
            f'soc={self.battery.soc/self.battery.capacity:.3} '
            f'dc={self.next_duty_cycle:.2}'
        ))

        return(
            self.battery.soc/self.battery.capacity, duty_cycle_real,
            e_in_real, e_out_real)

    def run(self, doys, e_ins):
        budget = np.zeros(len(doys))
        soc = np.zeros(len(doys))
        duty_cycle = np.zeros(len(doys))
        for i, (doy, e_in) in enumerate(zip(doys, e_ins)):
            soc[i], duty_cycle[i], e_in_real, budget[i] = self.step(
                doy, e_in)

        return soc, budget, duty_cycle


def effectiveness(doys, e_ins, e_outs, utility):

    e_ideal = utility(doys) / np.mean(utility(doys)) * np.mean(e_ins)
    return np.mean(np.minimum(e_ideal, e_outs)) / np.mean(e_ins)


def relative_underperformance(e_in, e_out, utility):

    e_tmp = e_out - (utility / np.mean(utility) * np.mean(e_in))
    e_tmp[e_tmp >= 0.0] = 0.0
    return - np.mean(e_tmp) / np.mean(e_in)
