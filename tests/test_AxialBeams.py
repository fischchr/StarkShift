import logging
from random import gauss
import unittest
import numpy as np
import pint

ureg = pint.UnitRegistry()

# Import the functions to test
from StarkShift.AxialBeams import BottleBeam, GaussianBeam, get_beam_power, get_P0


# Set up logging
logging.basicConfig(filename='tests/logs/test_AxialBeams.log', level=logging.INFO)

# GaussianBeam settings
P = 10 * ureg('mW')
w0 = 1 * ureg('um')
lam = 1064 * ureg('nm')
k = 2*np.pi / lam * np.array([0, 0, 1])
r0 = np.array([0, 0, 0]) * ureg('um')

# Additional BottleBeam settings
I0 = 250 * ureg('mW/um^2')
a = 0.62
NA = 0.4
d_max = 20 * ureg('um')
N_eval = 31

#axial = 
gaussian = GaussianBeam(P, k, w0, r0, ureg)
bob = BottleBeam(I0, k, a, NA, r0, ureg, d_max, N_eval)
 
# Define the testcase
class TestAxialBeams(unittest.TestCase):
    """Test the axial beam classes. """

    def test_P(self):
        """Test that the beam power is calculated correctly. """

        # Output of the function for a Gaussian beam
        P_test = get_beam_power(gaussian, 10*w0)
        # Analytic result
        P_calc = gaussian.P
        # Relative error
        rel_err = np.abs((P_test - P_calc) / P_calc).to_base_units().magnitude
        # Log results
        logging.debug(f'test_P: {P_test=}, {P_calc=}, {rel_err=}')

        # Verify result
        self.assertTrue(rel_err < 1e-6)

        # Check that the BottleBeam implementation works
        if N_eval >= 31:
            # Evaluate beam power
            P_bob = bob.P
            # The expected result for high N_eval
            P_calc = 585.3075 * ureg('mW')
            # Relative error
            rel_err = np.abs((P_bob - P_calc) / P_calc).to_base_units().magnitude
            # Log results
            logging.debug(f'test_P: {P_bob=}, {P_calc=}, {rel_err=}')

            # Verify result
            self.assertTrue(rel_err < 1e-1)
