# StarkShift

A library for calculating ac Stark shifts on alkali atoms, alkali Rydberg atoms, and alkaline-earth Rydberg atoms.

## Installation

The library can be installed using pip either from gitlab

```bash
$ pip3 install git+https://gitlab.phys.ethz.ch/tiqi-projects/optical-trap/StarkShift/
```

or after downloading the repository
```bash
$ git clone https://gitlab.phys.ethz.ch/tiqi-projects/optical-trap/StarkShift/
$ cd StarkShift && pip3 install ./
```

## Examples

The following examples show how the ac Stark shift can be calculated. 
Also check the unit tests in `tests/`, which show how the different functions are used.

### Axial Beams

The library is using `AxialBeam` objects to define laser beams. So far, a Gaussian beam

```python
import pint
import numpy as np
from StarkShift import GaussianBeam

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define a gaussian laser 
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Beam waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

beam = GaussianBeam(P, k, w0, r0, ureg)

# Get properties of the beam
print(beam.lam, beam.I0, beam.zR, beam.w0, beam.omega)
```

and a bottle-beam have been implemented

```python
from StarkShift import BottleBeam

# Define a bottle-beam 
I0 = 1 * ureg('W/cm^2')                 # Intensity scaling factor
lam = 532 * ureg('nm')                  # Wavelength
k = 2*np.pi * np.array([0, 0, 1]) / lam # k-vector
r0 = np.array([1, 0, 1]) * ureg('um')   # Position of the focus

a = 0.62 # scaled phase-inversion radius
NA = 0.4 # NA of the imaging system

bob = BottleBeam(I0, k, a, NA, r0, u, N_eval=51)
```

### Expansion of the beam

For the calculation of ac Stark shifts on Rydberg states, the class `SphericalBeamExpansion` is used, which uses the expansion of a function into spherical harmonics implemented by [pyshtools](https://shtools.github.io/SHTOOLS/index.html).
The following example shows, how a beam can be expanded into spherical harmonics:

```python
# Assuming an axial beam has been initialized as shown above
from StarkShift import SphericalBeamExpansion

# Expand the beam in spherical harmonics
beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o =2 * w0)
```

### ac Stark shift on an alkali or alkaline-earth atom

The ac Stark shift on an alkali or alkaline-earth atom in a low-lying electronic state is calculated in the following example (see `examples/example-1_alkali_atom.py`):

```python
import pint 
import numpy as np
from atomphys import Atom

from StarkShift import alkali_core_ac_stark
from StarkShift import GaussianBeam
from StarkShift import alkali_core_polarizability, alkali_core_ac_stark

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the atomphys atom
ap_atom = Atom('Ca', ureg=ureg)
# Define the atomphys state
ap_state = ap_atom('1S0')
# Define the mj value
mj1 = 0

# Define the laser properties
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Beam waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

# Define the laser beam
beam = GaussianBeam(P, k, w0, r0, ureg)

# Define the polarization
epsilon = 'x'

# Calculate the polarizability
alpha = alkali_core_polarizability(ap_state, mj1, beam, epsilon)

# Or calculate the ac Stark shift
U = alkali_core_ac_stark(ap_state, mj1, beam, epsilon)

# Get the ac Stark shift in mK
U_kB = (U / ureg('k_B')).to('mK')
```

Note that the values calculated here are identical to the results obtained with [atomphys](https://atomphys.org/).
For calculating a large number of polarizabilities, it is recommended to use the corresponding function in `atomphys`.

### ac Stark shift on an alkali Rydberg atom

The ac Stark shift on a alkali Rydberg atom is calculated in the following example (see `examples/example-2_alkali_rydberg.py`):

```python
import pint 
import numpy as np
from arc import Potassium

from StarkShift import GaussianBeam, SphericalBeamExpansion
from StarkShift import alkali_rydberg_polarizability, alkali_rydberg_ac_stark

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the ARC atom
arc_atom = Potassium()
# Define the Rydberg state (n, s, l, j, mj)
state_r = (50, 1/2, 0, 1/2, 1/2)

# Define the laser properties
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Beam waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

# Define the laser beam
beam = GaussianBeam(P, k, w0, r0, ureg)

# Expand the beam in spherical harmonics
beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o =2 * w0)

# Calculate the polarizability
alpha = alkali_rydberg_polarizability(state_r, beam_expansion, arc_atom)

# Or calculate the ac Stark shift
U = alkali_rydberg_ac_stark(state_r, beam_expansion, arc_atom)

# Get the ac Stark shift in mK
U_kB = (U / ureg('k_B')).to('mK')

print(U_kB)
```
