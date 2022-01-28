import pint
import logging
import unittest
import numpy as np
from arc import Hydrogen
from atomphys import Atom
from StarkShift.AxialBeams import BottleBeam


# Import the functions to test
from StarkShift.StarkShiftUtil import find_trap_depth

# Set up logging
logging.basicConfig(filename='tests/logs/test_StarkShiftUtil.log', level=logging.INFO)

# Define the test case

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the atomphys atom
ap_atom = Atom('Ca+', ureg=ureg)
# Define the core state
state_c = ap_atom('S1/2')
#configuration_c = (1/2, 'S1/2')

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
dmax = 6 * ureg('um')

bob = BottleBeam(P, k, a, NA, r0, ureg, dmax, 101)

# Define the polarization of the laser relative to it's k-vector
epsilon = 'x'

# Define the testcase
class TestStarkShiftUtil(unittest.TestCase):
    """Test Stark shift utilities. """

    def test_find_trap_depth(self):
        """Test the function which finds the trap depth. """

        r_saddle, phi_saddle, U_saddle, trap_depth = find_trap_depth(state_c, arc_atom, state_r, j, mj, bob, epsilon)
        logging.info(f'find_trap_depth: {r_saddle=}, {phi_saddle=}, {U_saddle=}, {trap_depth=}')