import pint 
import unittest
import logging
import numpy as np
from arc import Potassium
from atomphys import Atom

# Import the functions to test
from StarkShift import alkali_core_ac_stark
from StarkShift import alkali_rydberg_ac_stark
from StarkShift import GaussianBeam, SphericalBeamExpansion
from StarkShift import alkaline_earth_core_ac_stark, alkaline_earth_rydberg_ac_stark, alkaline_earth_ac_stark

# Set up logging
logging.basicConfig(filename='tests/logs/test_AlkalineEarthPolarizability.log', level=logging.INFO)

# Initialize unit registry
ureg = pint.UnitRegistry()

# Define the laser properties
P = 100 * ureg('mW')                    # Beam power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Beam waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

# Define the beam
beam = GaussianBeam(P, k, w0, r0, ureg)

# Expand the beam in spherical harmonics
beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o =2 * w0)

# Define the polarization
epsilon = 'x'
theta_p = np.pi / 2
    

# Define the testcase
class AlkalineEarthPolarizability(unittest.TestCase):
    """Test the calculation of the polarizability of a Rydberg state of an alkaline-earth atom. """

    # Test cases
    def test_alkaline_earth_core_ac_stark(self):
        """Check that `alkaline_earth_core_ac_stark` gives the same result as the alkali atom function when l2=s2=0. """

        # Define the atomphys atom
        ap_atom = Atom('Ca+', ureg=ureg)
        # Define the core state
        state_c = ap_atom('S1/2')

        # Define the ARC atom (for testing it doesn't matter that it's not the same as the atom phys atom)
        arc_atom = Potassium()
        # Define the Rydberg state
        state_r = (0, 0, 0, 0, 0)

        # Define the total angular momentum
        j = 1/2
        mj = j

        # Calculate the ac Stark shift using the alkaline-earth atom function
        U_AE = alkaline_earth_core_ac_stark(state_c, state_r, j, mj, epsilon, beam)

        # Calculate the ac Stark shift using the alkali atom function
        U_A = alkali_core_ac_stark(state_c, mj, beam.I0, beam.omega, epsilon, ureg)

        # Calculate the difference
        diff = U_AE - U_A
        rel_err = np.abs(diff / U_A).to_base_units().magnitude

        # Log the result
        line1 = 'test_alkaline_earth_core_ac_stark. \n'
        line2 = f'\t{state_c=}, {state_r=}, {j=}, {mj=}\n'
        line3 = f"\t{U_AE.to('eV')=}, {U_A.to('eV')=}, {diff=}, {rel_err=}"
        logging.info(line1 + line2 + line3)

        assert diff < 1e-10 * ureg('eV')

    def test_alkaline_earth_rydberg_ac_stark(self):
        """Check that `alkaline_earth_rydberg_ac_stark` gives the same result as the alkali atom function when j1=0. """

        # Define the atomphys atom. We only need this to get an atomphys.state object
        ap_atom = Atom('Ca', ureg=ureg)
        # Define the core state with j1 = 0
        state_c = ap_atom('1S0')

        # Define the ARC atom (for testing it doesn't matter that it's not the same as the atom phys atom)
        arc_atom = Potassium()
        # Define the Rydberg state
        state_r = (50, 0, 49, 49, 49)

        # Define the total angular momentum
        j = 49
        mj = j

        # Calculate the ac Stark shift using the alkaline-earth atom function
        U_AE = alkaline_earth_rydberg_ac_stark(state_c, state_r, j, mj, beam_expansion, arc_atom)

        # Calculate the ac Stark shift using the alkali atom function
        U_A = alkali_rydberg_ac_stark(state_r, beam_expansion, arc_atom)

        # Calculate the difference
        diff = U_AE - U_A
        rel_err = np.abs(diff / U_A).to_base_units().magnitude

        # Log the result
        line1 = 'test_alkaline_earth_rydberg_ac_stark. \n'
        line2 = f'\t{state_c=}, {state_r=}, {j=}, {mj=}\n'
        line3 = f"\t{U_AE.to('eV')=}, {U_A.to('eV')=}, {diff=}, {rel_err=}"
        logging.info(line1 + line2 + line3)

        assert diff < 1e-10 * ureg('eV')

    def test_alkaline_earth_ac_stark(self):
        """Calculate the total ac Stark shift. """

        # Define the atomphys atom
        ap_atom = Atom('Ca+', ureg=ureg)
        # Define the core state
        state_c = ap_atom('S1/2')

        # Define the ARC atom (for testing it doesn't matter that it's not the same as the atom phys atom)
        arc_atom = Potassium()
        # Define the Rydberg state
        state_r = (50, 0, 49, 49, 49)

        # Combined angular momentum
        j = 49 + 1/2
        mj = j

        U = alkaline_earth_ac_stark(state_c, state_r, j, mj, beam, epsilon, arc_atom=arc_atom)

        # Log the result
        line1 = 'test_alkaline_earth_ac_stark. \n'
        line2 = f'\t{state_c=}, {state_r=}, {j=}, {mj=}\n'
        line3 = f"\t{U.to('eV')=}"
        logging.info(line1 + line2 + line3)


if __name__ == '__main__':
    # Run the tests
    unittest.main()

