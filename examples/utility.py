import enmanage
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # enmanage package contains example data from 5 locations.
    # here, we load the "berlin" dataset
    solar_data = enmanage.data["C_BLN"]

    doys = np.array(solar_data["data"]["exposure"].index.dayofyear)
    e_ins = solar_data["data"]["exposure"].values

    # arguments passed to the predictor. we train of the first year of data.
    predictor_args = {
        "training_data": {"doy": doys[:365], "e_in": e_ins[:365]},
        "latitude": solar_data["latitude"],
        "window_size": 63,
    }

    sim_args = {"battery": {"capacity_mah": 15000}}
    f, axarr = plt.subplots(2, 1, sharex=True)
    # we evaluate two utility profiles:
    for utility in ["uniform", "seasonal"]:
        # arguments passed to the em algorithm. we want a uniform profile and some specific
        # parameters for preact's PID controller.
        manager_args = {
            "utility_function": enmanage.profiles[utility],
            "control_coefficients": {"k_p": 1.5, "k_i": 0.00152, "k_d": 0.0},
        }

        simulator = enmanage.Simulator.from_config(
            enmanage.PREACT,
            manager_args=manager_args,
            predictor_args=predictor_args,
            config_dict=sim_args,
        )
        # evaluate over 4 years
        soc, budget, duty_cycle = simulator.run(
            doys[365 : 5 * 365], e_ins[365 : 5 * 365]
        )

        axarr[0].plot(soc)
        axarr[1].plot(duty_cycle, label=f"Utility '{utility}'")
    axarr[0].set_title("State of charge")
    axarr[1].set_title("Duty-cycle")
    axarr[1].legend()
    plt.show()
