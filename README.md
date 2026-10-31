# Example scripts

Scripts that compare the time-local dynamical maps against brute-force ACE simulations for a two-level system with phonons.

| script | content |
|---|---|
| `spectrum_cw.py` | CW excitation: G1 and spectrum |
| `spectrum_pulsed.py` | spectrum under pulsed excitation |
| `purity_pulse_area.py` | single-photon purity against the pulse area |

## Requirements

Linux with Python >= 3.10, a C++ and a Fortran compiler, CMake, OpenBLAS and Eigen. On Debian/Ubuntu:

```bash
sudo apt install build-essential gfortran cmake libopenblas-dev libeigen3-dev python3-venv git
```

Tested on Ubuntu 24.04 with Python 3.12. Should also work on WSL.

## Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Clone and install ACE (branch `gil_nanobind`):

   ```bash
   git clone -b gil_nanobind https://github.com/tbracht/ACE.git
   pip install ./ACE
   ```

3. Clone and install pyaceqd (branch `ace_update`):

   ```bash
   git clone -b ace_update https://github.com/tbracht/pyaceqd.git
   pip install ./pyaceqd
   ```

## Running

With the virtual environment active, run a script from the folder that contains it, for example:

```bash
python spectrum_cw_detector.py
```

Each script prints a comparison of the time-local and brute-force results and saves a figure (`.png`) and the data (`.npz`) in the current folder. On the first run the phonon process tensor is computed and stored in the current folder as `tls_5.0nm_4.0k_th8_dt0.1.pt_*`, later runs reuse it.
