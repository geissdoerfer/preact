import numpy as np
import matplotlib as mpl
from scipy import interpolate
import matplotlib.pyplot as plt

wl = 13
margin = int(np.floor(wl / 2))
wndw = np.ones(wl) / wl

ds = np.arange(365)

p = dict()
p["uniform"] = np.ones(365)

rnd_holidays = np.ones(365) * 0.5

holidays = list()
# 01.01.-22.01.: 1-22
holidays += range(0, 22)
# 26.01.: 26
holidays += [25]
# 01.04.17.04.: 91-107
holidays += range(90, 107)
# 25.04.: 115
holidays += [114]
# 01.05.: 121
holidays += [120]
# 24.06. - 09.07.: 175-190
holidays += range(174, 190)
# 16.09. - 02.10.: 259-275
holidays += range(258, 275)
# 16.10. 289
holidays += [288]
# 09.12. - 31.12. 343-365
holidays += range(342, 365)

weekends = [i for i in range(7, 365, 7)]
weekends += [i for i in range(8, 365, 7)]

rnd_holidays[holidays] = 1.0
rnd_holidays[weekends] = 1.0

p["holidays"] = rnd_holidays

p_tmp = np.ones(365)
p_tmp[3 * 30 : 4 * 30] = np.linspace(1.0, 0.1, 30)
p_tmp[4 * 30 : 8 * 30] = 0.1
p_tmp[8 * 30 : 9 * 30] = np.linspace(0.1, 1.0, 30)

p_seasonal = np.convolve(p_tmp, wndw, "full")[margin:-margin]
p_seasonal[:wl] = 1.0
p_seasonal[-wl:] = 1.0

p["seasonal"] = p_seasonal

profiles = dict()


def create_profile_function(profile):
    def fun(d):
        return p[profile][d % 365]

    return fun


for profile in p.keys():
    profiles[profile] = create_profile_function(profile)
