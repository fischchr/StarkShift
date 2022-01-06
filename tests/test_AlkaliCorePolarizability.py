import pint 
import unittest
import logging
import numpy as np
from atomphys.state import State
from atomphys import Atom, Laser

# Import the functions to test
from StarkShift import GaussianBeam
from StarkShift import alkali_core_polarizability

# Set up logging
logging.basicConfig(filename='tests/logs/test_AlkaliCorePolarizability.log', level=logging.INFO)


# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the atomphys atom
ap_atom = Atom('Ca+', ureg=ureg)


# Define the laser properties
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Beam waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

# Define the laser beam
beam = GaussianBeam(P, k, w0, r0, ureg)

# Define the atomphys laser
ap_laser = Laser(units=ureg, λ=np.linspace(lam.to('nm').magnitude, lam.to('nm').magnitude + 1, 1) * ureg.nm, A=0)



# Define the testcase
class TestAlkaliCorePolarizability(unittest.TestCase):
    """Test the calculation of the polarizability against `atomphys`. """

    def setUp(self):
        """Define the atomic configurations to test. """

        self.test_configurations = [('S', 1/2), ('P', 1/2), ('P', 3/2), ('D', 3/2), ('D', 5/2)]

    # Helper functions
    def _compare_polarizability(self, ap_state: State, j1: float, epsilon: str, theta_p: float, theta_k: float):
        """Function for comparing the polarizability of all mj states of one configuration. """

        # Iterate over all possible values of mj
        for mj1 in [1/2 + i for i in range(int(j1 + 1/2))]:
            # Calculate the polarizability using atomphys
            alpha_ap = ap_state.α(mJ=mj1, laser=ap_laser, theta_k=theta_k, theta_p=theta_p)[0].to('e * a_u_length / a_u_electric_field')
            # Calculate the polarizability using StarkShift
            alpha = alkali_core_polarizability(ap_state, mj1, beam, epsilon)
            # Calculate the difference
            diff = np.abs(alpha - alpha_ap)

            # Log the result
            line1 = '_compare_polarizability. \n'
            line2 = f'\t{ap_state=}\n'
            line3 = f'\t{epsilon=}, {theta_p=}, {theta_p=}\n'
            line4 = f'\t{alpha=:.2f}, {alpha_ap=:.2f}, {diff=}'
            logging.info(line1 + line2 + line3 + line4)

            # Check the difference
            self.assertTrue(diff < 1e-3 * ureg('e * a_0 / a_u_electric_field'))

    # Test cases
    def test_alkali_core_polarizability_z_polarization(self):
        """Test `alkali_core_polarizability` for z polarization (theta_p = 0). """

        # Define the polarization
        epsilon = 'z'
        theta_p = 0

        # Iterate over all configurations to test
        for (l1, j1) in self.test_configurations:
            # Get the label of the state
            state = f'{l1}{2*j1:.0f}/2'
            # Get the corresponding atom phys state
            ap_state = ap_atom(state)
            # Compare the polarizability
            self._compare_polarizability(ap_state, j1, epsilon, theta_p, theta_k=np.pi/2)

    def test_states_x_polarization(self):
        """Test `alkali_core_polarizability` for x polarization (theta_p = pi/2). """

        # Define the polarization
        epsilon = 'x'
        theta_p = np.pi / 2

        # Iterate over all configurations to test
        for (l1, j1) in self.test_configurations:
            # Get the label of the state
            state = f'{l1}{2*j1:.0f}/2'
            # Get the corresponding atom phys state
            ap_state = ap_atom(state)
            # Compare the polarizability
            self._compare_polarizability(ap_state, j1, epsilon, theta_p, theta_k=np.pi/2)


if __name__ == '__main__':
    # Run the tests
    unittest.main()

