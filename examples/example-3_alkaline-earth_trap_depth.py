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

print(trap_depth.to('mK'))