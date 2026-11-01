# Single-photon purity of a TLS with phonons against the pulse area: time-local maps (use_dm=True) on a dense grid,
# brute force (use_dm=False) on a subset of it. 1 - P = 2 int_0^tb/2 G2 / int_tb/2^3tb/2 G2, see Indistinguishability.calc_purity.
# Results are saved to purity_pulse_area.npz and loaded from there if the file exists (delete it to recompute).
# usage: python purity_pulse_area.py
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.two_level_system.tls import TLS
from pyaceqd.two_time.purity import Indistinguishability
from pyaceqd.helpers.ace_operators import ketbra

tau_0 = 5            # pulse duration in ps
t_center = 4.5*tau_0
tb = 2000            # time between two pulses of the train
max_pulse_t = 2*t_center  # covers the pulse envelope
t_mem = 20
areas = np.round(np.linspace(0.1, 10, 100), 5)  # pulse area in units of pi
areas_bf = areas[9::10]                         # 1, 2, ..., 10: brute force is much slower

tls = TLS(dt=0.1, gamma_e=1/100, lindblad=True, phonons=True, ae=5, temperature=4, verbose=False)


def purity(area, use_dm):
    laser = ChirpedPulse(tau_0=tau_0, detuning=0, t_center=t_center, pulse_area=area, polarization="x", repeat_tb=tb)
    indist = Indistinguishability(tls, ketbra(0, 1, 2), laser, tb=tb, dt_small=0.1, dt_big=1, max_pulse_t=max_pulse_t,
                                  t_mem=t_mem, use_dm=use_dm)
    return indist.calc_purity()


def sweep(areas, use_dm):
    P = np.zeros(len(areas))
    start = time.time()
    for i, area in enumerate(areas):
        P[i] = purity(area, use_dm)
        print("{}: area {:.2f} pi, 1 - P = {:.4e}".format("tl" if use_dm else "bf", area, 1 - P[i]))
    print("{} sweep: {:.1f} s".format("time-local" if use_dm else "brute force", time.time() - start))
    return P


data_file = "purity_pulse_area.npz"
if os.path.exists(data_file):
    print("loading", data_file)
    data = np.load(data_file)
    areas, P_tl, areas_bf, P_bf = data["areas"], data["P_tl"], data["areas_bf"], data["P_bf"]
else:
    P_tl = sweep(areas, use_dm=True)
    P_bf = sweep(areas_bf, use_dm=False)
    np.savez(data_file, areas=areas, P_tl=P_tl, areas_bf=areas_bf, P_bf=P_bf)
P_tl_at_bf = P_tl[np.searchsorted(areas, areas_bf)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(areas, 1 - P_tl, "k-", lw=1, label="tl")
ax.plot(areas_bf, 1 - P_bf, "bo", alpha=0.5, label="bf")
ax.set_xlabel("pulse area in units of pi")
ax.set_ylabel("1 - P")
ax.legend()
fig.tight_layout()
fig.savefig("purity_pulse_area.png", dpi=120)
print("saved purity_pulse_area.png, purity_pulse_area.npz")
