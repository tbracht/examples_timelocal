# Spectrum of a TLS with phonons under pulsed excitation: time-local maps (dm=True) against brute force (dm=False).
# S(E) = Re int_0^tend dtau exp(-i E tau/hbar) int dt G1(t, tau), see Spectrum.get_onesided_spectrum.
# usage: python spectrum_pulsed.py
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyaceqd.pulses import ChirpedPulse
from pyaceqd.two_level_system.tls import TLS
from pyaceqd.two_time.spectrum import Spectrum
from pyaceqd.helpers.ace_operators import ketbra

tau_0 = 3           # pulse duration in ps
t_center = 5*tau_0
pulse_area = 6      # pulse area in units of pi
tend = 4000         # maximum t and tau
t_mem = 20
max_pulse_t = 30    # covers the pulse envelope

laser = ChirpedPulse(tau_0=tau_0, detuning=0, t_center=t_center, pulse_area=pulse_area, polarization="x")
tls = TLS(dt=0.1, gamma_e=1/100, lindblad=True, phonons=True, ae=5, temperature=4, verbose=False)
spec = Spectrum(tls, ketbra(0,1,2), ketbra(1,0,2), laser,
                tend=tend, dt_small=1, dt_big=1, max_pulse_t=max_pulse_t, t_mem=t_mem, verbose=True, workers=15)

start = time.time()
energies, S_tl = spec.get_onesided_spectrum(e_min=-5, e_max=5, dm=True)
t_tl = time.time() - start
start = time.time()
_, S_bf = spec.get_onesided_spectrum(e_min=-5, e_max=5, dm=False)
t_bf = time.time() - start
print("time-local {:.1f} s, brute force {:.1f} s".format(t_tl, t_bf))
np.savez("spectrum_pulsed.npz", energies=energies, S_tl=S_tl, S_bf=S_bf)

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(energies, np.abs(S_bf), "b-", lw=3, alpha=0.4, label="bf")
ax.semilogy(energies, np.abs(S_tl), "k--", lw=1, label="tl")
ax.set_xlabel("E in meV")
ax.set_ylabel("|S(E)|")
ax.legend(loc="upper left")
ax2 = ax.twinx()
ax2.semilogy(energies, np.abs(S_tl - S_bf)/np.abs(S_bf), "r-", lw=0.8)
ax2.set_ylabel("|tl - bf| / |bf|", color="r")
fig.tight_layout()
fig.savefig("spectrum_pulsed.png", dpi=120)
print("saved spectrum_pulsed.png, spectrum_pulsed.npz")
