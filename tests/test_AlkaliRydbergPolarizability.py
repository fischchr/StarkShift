import pint 
import unittest
import logging
import numpy as np
from arc import Rubidium

# Import the functions to test
from StarkShift import alkali_rydberg_polarizability
from StarkShift import GaussianBeam, SphericalBeamExpansion

# Set up logging
logging.basicConfig(filename='tests/logs/test_AlkaliRydbergPolarizability.log', level=logging.INFO)

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the ARC atom
arc_atom = Rubidium()

# Define the laser properties
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist


# Define the testcase
class TestAlkaliRydbergPolarizability(unittest.TestCase):
    """Test the calculation of the polarizability of a Rydberg state. """

    def setUp(self):

        # Define the configurations we want to test (n, s, l, j)
        self.configurations = [
            (50, 1/2, 0, 1/2),
            (50, 0, 2, 2),
        ]

    # Helper functions
    def _compare_polarizability(self, configuration: tuple, beam_expansion: SphericalBeamExpansion, rel_err_tol: float):
        """Function for comparing the polarizability of all mj states of one configuration. """

        # Get the configuration
        n2, s2, l2, j2 = configuration

        # Get the beam waist
        w0 = beam_expansion.beam.w0
        # Get the beam intensity
        I0 = beam_expansion.beam.I0
        # Get the wavelength
        lam = beam_expansion.beam.lam

        # Calculate polarizability of a free electron
        omega = 2 * np.pi * ureg('c') / beam_expansion.beam.lam
        alpha_p = -1 * ureg('e')**2 / (ureg('m_e') * omega**2)
        alpha_p.ito('e * a_0 / a_u_electric_field')

        # Iterate over all possible values of mj
        for mj2 in [(-j2 + i) for i in range(int(2*j2) + 1)]:
            state = (n2, s2, l2, j2, mj2)
            alpha = alkali_rydberg_polarizability(state, beam_expansion, arc_atom)

            # Get the difference to a free electron
            diff = alpha - alpha_p
            # Get the relative error
            rel_err = np.abs(diff / alpha).to_base_units().magnitude
            
            # Log the result
            line1 = '_compare_polarizability. \n'
            line2 = f'\t{state=}\n'
            line3 = f'\t{w0=}, {I0=}, {lam=}\n'
            line4 = f'\t{alpha=:.2f}, {alpha_p=:.2f}, {diff=}, {rel_err=}'
            logging.info(line1 + line2 + line3 + line4)

            # Make sure the polarizability is negative
            self.assertTrue(alpha < 0 * ureg('e * a0 / a_u_electric_field'))

            # Check the difference
            self.assertTrue(rel_err < rel_err_tol)

            # Make sure the magnitude of the polarizability is smaller than the free case 
            # because the beam intensity of the Gaussian beam is falling off to the sides compared to a plane wave.
            self.assertTrue(alpha_p < alpha)   

    # Test cases
    def test_alkali_rydberg_polarizability_large_beam(self):
        """Test `alkali_rydberg_polarizability` for a large beam where the beam can be approximated to be constant over the extent of the Rydberg wave function. """

        # Define the waist
        w0 = 10 * ureg('um')
        rel_err_tol = 0.01

        # Define the beam
        beam = GaussianBeam(P, k, w0, r0, ureg)

        # Expand the beam in spherical harmonics
        beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o=2 * w0)

        for configuration in self.configurations:
            self._compare_polarizability(configuration, beam_expansion, rel_err_tol)

    def test_alkali_rydberg_polarizability_small_beam(self):
        """Test `alkali_rydberg_polarizability` for a small beam where the polarizability is significantly lower than for a plane wave. """

        # Define the waist
        w0 = 1 * ureg('um')
        rel_err_tol = 0.1

        # Define the beam
        beam = GaussianBeam(P, k, w0, r0, ureg)

        # Expand the beam in spherical harmonics
        beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o=2 * w0)

        for configuration in self.configurations:
            self._compare_polarizability(configuration, beam_expansion, rel_err_tol)


if __name__ == '__main__':
    # Run the tests
    unittest.main()

