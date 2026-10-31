# CW detector signal of a TLS with phonons under weak driving: G1 and the spectrum seen with a Lorentzian detector,
# brute force (SpectrumCW.G1) and time-local maps (SpectrumCW.G1_tl). Same parameters as spectrum_cw_analytical.py,
# see spectrum_cw_weak.py for the incoherent spectrum.
# The phonon runs lose a little trace (~1e-3 over the 6000 ps relaxation, threshold 8). SpectrumCW divides it out
# (brute force: by Tr rho(tau) of the run without MTO, time-local: by lambda^k of the stationary map), this changes the
# spectrum by ~1e-5 relative compared to normalising by Tr rho(0) only as in spectrum_cw_analytical.py.
# usage: python spectrum_cw_detector.py
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyaceqd.pulses import CWLaser
from pyaceqd.two_level_system.tls import TLS
from pyaceqd.two_time.spectrum import onesided_ft
from pyaceqd.two_time.spectrum_cw import SpectrumCW
from pyaceqd.helpers.ace_operators import op_to_matrix
from pyaceqd.constants import hbar

rabi_energy = 0.001  # driving strength hbar*Omega in meV, H = -hbar*Omega/2 (|1><0| + |0><1|)
detuning = 0.0
gamma_e = 0.01      # radiative decay in 1/ps
gamma_det = 0.01    # detector: G1 times exp(-gamma_det tau/hbar), i.e. a Lorentzian filter with HWHM gamma_det in meV
tend = 3000
t_relax = 6000      # relaxation to the steady state from -t_relax to t = 0 (brute force)
t_mem = 20


def detector_spectrum(tau, G1, energies):
    """S(E) = Re int_0^tend G1(tau) exp(-(i E + gamma_det) tau / hbar) with the trapezoidal rule, including the coherent part"""
    return np.real(onesided_ft(G1*np.exp(-gamma_det*tau/hbar), tau, energies))


tls = TLS(dt=0.1, gamma_e=gamma_e, lindblad=True, phonons=True, ae=5, temperature=4, threshold=8, verbose=False)
# the system multiplies the pulse amplitude by pi, so a CW amplitude e0 gives Omega = pi e0
laser = CWLaser(rabi_energy/(np.pi*hbar), detuning=detuning, polarization="x")
spec = SpectrumCW(tls, op_to_matrix("|0><1|_2"), laser, tend=tend, t_relax=t_relax, t_mem=t_mem)
start = time.time()
tau, G1_bf, _ = spec.G1()
t_bf = time.time() - start
start = time.time()
_, G1_tl, _ = spec.G1_tl()
t_tl = time.time() - start
print("brute force {:.1f} s, time-local {:.1f} s".format(t_bf, t_tl))
print("<x>_ss = G1(0): bf {:.10f}, tl {:.10f}".format(G1_bf[0].real, G1_tl[0].real))
print("max |G1_tl - G1_bf| / |G1(0)| = {:.2e}".format(np.max(np.abs(G1_tl - G1_bf))/abs(G1_bf[0])))

energies = np.linspace(-4, 4, 2001)
S_bf, S_tl = detector_spectrum(tau, G1_bf, energies), detector_spectrum(tau, G1_tl, energies)
print("detector spectrum: max |S_tl - S_bf| / S_bf = {:.2e}".format(np.max(np.abs(S_tl - S_bf)/np.abs(S_bf))))
i_m, i_p = np.argmin(np.abs(energies + 1)), np.argmin(np.abs(energies - 1))
print("phonon sideband: S(-1 meV) = {:.3e}, S(+1 meV) = {:.3e}, ratio -/+ = {:.2f}".format(S_bf[i_m], S_bf[i_p], S_bf[i_m]/S_bf[i_p]))
np.savez("spectrum_cw_detector.npz", tau=tau, G1_bf=G1_bf, G1_tl=G1_tl, energies=energies, S_bf=S_bf, S_tl=S_tl)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(tau, G1_bf.real, "b-", lw=3, alpha=0.4, label="bf real")
ax.plot(tau, G1_tl.real, "k--", lw=1, label="tl real")
ax.plot(tau, G1_bf.imag, "r-", lw=3, alpha=0.4, label="bf imag")
ax.plot(tau, G1_tl.imag, "k:", lw=1, label="tl imag")
ax.set_xlim(0, 600)
ax.set_xlabel("tau in ps")
ax.set_title("G1")
ax.legend(fontsize=7)
ax = axes[1]
ax.semilogy(energies, np.abs(S_bf), "b-", lw=3, alpha=0.4, label="bf")
ax.semilogy(energies, np.abs(S_tl), "k--", lw=1, label="tl")
ax.semilogy(energies, np.abs(S_tl - S_bf) + 1e-16, "r-", lw=0.8, label="|tl - bf|")
ax.axvline(detuning, color="0.5", ls=":", label="laser")
ax.set_xlabel("E in meV")
ax.set_title("spectrum, detector HWHM {} meV".format(gamma_det))
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("spectrum_cw_detector.png", dpi=120)
print("saved spectrum_cw_detector.png, spectrum_cw_detector.npz")
