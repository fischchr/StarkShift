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

## Atomic states

The library is using internally the [atomphys](https://github.com/mgrau/atomphys) and [ARC](https://github.com/nikolasibalic/ARC-Alkali-Rydberg-Calculator) library.
Therefore, different notations are used to define the state of an electron.

For low-lying electronic states, `atomphy` is used and a state might be defined as

```python
from atomphys import Atom

# Define the atomphys atom
ap_atom = Atom('Ca')
# Define the atomphys state
ap_state = ap_atom('1S0')
# Define the mj value (ap_state only knows the total angular momentum j)
mj1 = 0
```

For Rydberg states, `arc` is used and a state is defined by its quantum numbers

```python
from arc import Potassium

# Define the ARC atom
arc_atom = Potassium()
# Define the Rydberg state (n, s, l, j, mj)
state_r = (50, 1/2, 0, 1/2, 1/2)
```

Finally, for alkaline-earth Rydberg states, a combination of the two is used

```python
from atomphys import Atom
from arc import Calcium40

# Define the atomphys atom for the core state
ap_atom = Atom('Ca+')
# Define the core state
ap_state = ap_atom('1S1/2')

# Define the ARC atom
arc_atom = Calcium40()
# Define the Rydberg state (n2, s2, l2, j2)
# Note that you can either set s2 = 1/2 (jj coupling) or s2 = 0 (jK coupling)
state_r = (50, 0, 49, 49)

# Define the total angular momenutm (in jK coupling in this case)
j = 49 + 1/2
mj = j
```

## Laser beams

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

and a bottle-beam

```python
from StarkShift import BottleBeam

# Define a bottle-beam 
P = 500 * ureg('mW')                    # Total power of the bottle beam
lam = 532 * ureg('nm')                  # Wavelength
k = 2*np.pi * np.array([0, 0, 1]) / lam # k-vector
r0 = np.array([1, 0, 1]) * ureg('um')   # Position of the focus

a = 0.62 # scaled phase-inversion radius
NA = 0.4 # NA of the imaging system

bob = BottleBeam(P, k, a, NA, r0, u, N_eval=51)
```

have been implemented.

## Quantization axis and polarization

By default, the quantization axis of the atom is defined by the k-vector of the laser beam object from above (a different quantization axis can be passed in as a keyword argument, see e.g., `alkali_core_ac_stark`).
The polarization vector is then defined with respect to this quantization axis.
For example, a beam propagating along z (k = |k| e_z) with linear polarization has a polarization vector epsilon = cos(ϕ) e_x + sin(ϕ) e_y.
In the calculation, the polarization can be specified in text form, i.e., for the from before with linear polarization (ϕ = π/2)

```python
# Define a linear polarization
epsilon = 'cos(pi/4) * x + sin(pi/4) * y'
```

Similarly, when the beam is right or left circularly polarized,

```python
# Define a circular polarization
epsilon_rcp = 'x + iy'
epsilon_lcp = 'x - iy'
```

Note, that it is possible to define unphysical polarizations. For example, when k = |k| e_z, a value `epsilon = z` would correspond to an (unphysical) longitudinal polarization of the laser.

## Testing and verification

The calculation of Stark shifts has been tested and a set of unit tests is available in `tests/`.
The tests can be run by executing the script `run_tests.py`.
It is also possible to run idividual tests by invoking `python3 -m unittest tests/test_TESTCASE.py` where `TESTCASE` is the name of one of the unit tests.

The following tests have been performed:

* For low-lying electronic states the code has been compared against the `atomphys` library.
* For Rydberg states of alkali atoms, the calculation is compared against the polarizability of a free electron (the plane wave limit).
* For alkaline-earth Rydberg states, it was verified that the calculation results in the correct single-electron results when setting the quantum numbers of one of the two electrons to zero.

## Examples

The following examples show how the ac Stark shift can be calculated.
Also check the unit tests in `tests/`, which show how the different functions are used.

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

### Trap depth of a bottle beam

The exmple above shows how the trap depth of a bottle beam can be calculated for an alkaline-earth atom (see also `examples/example-3_alkaline-earth_trap_depth.py`).

```python
import pint
import numpy as np
from arc import Hydrogen
from atomphys import Atom
from StarkShift.AxialBeams import BottleBeam
from StarkShift.StarkShiftUtil import find_trap_depth


# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the atomphys atom
ap_atom = Atom('Ca+', ureg=ureg)
# Define the core state
state_c = ap_atom('S1/2')

# Define the ARC atom
arc_atom = Hydrogen()
# Define the Rydberg configuration (n, s, l, j)
state_r = (50, 0, 49, 49)

# Define the total angular momenutm
j = state_r[-1] + 1/2
mj = j

# Define the laser properties
P = 500 * ureg('mW')          # Beam power
k_hat = np.array([0, 0, 1])   # Direction of the k-vector
lam = 1183 * ureg('nm')       # Wavelength
k = 2*np.pi / lam * k_hat     # k-vector
NA = 0.4                      # Numerical aperture
a = 0.62                      # Phase-shifted area
r0 = np.zeros(3) * ureg('um') # Location of the focus
dmax = 6 * ureg('um')         # Region for interpolating the bottle beam

bob = BottleBeam(P, k, a, NA, r0, ureg, dmax, 101)

# Define the polarization of the laser relative to it's k-vector
epsilon = 'x'

r_saddle, phi_saddle, U_saddle, trap_depth = find_trap_depth(state_c, arc_atom, state_r, j, mj, bob, epsilon)
```
