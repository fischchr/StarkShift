import logging
import unittest
import numpy as np

# Import the functions to test
from StarkShift.PolarizationUtil import cart_to_sph, sph_to_cart, evaluate_vector_description

# Set up logging
logging.basicConfig(filename='tests/logs/test_PolarizationUtil.log', level=logging.INFO)


# Define the testcase
class TestPolarizationUtil(unittest.TestCase):
    """Test the conversion of strings to polarization vectors. """

    # Helper functions
    def _check_difference(self, vec_1: np.ndarray, vec_2: np.ndarray, eps: float = 1e-12):
        """Check that the difference between two vectors is small. """

        # Calculate the difference
        diff = vec_1 - vec_2
        # Calculate the magntiude
        mag = np.sum(diff * diff.conj())
        # Check that the magnitude is small
        self.assertTrue(mag < eps)

    def _check_cart(self, polarization: str, res_xyz: np.ndarray):
        """Check the evaluation in caresian coordinates. """

        eps_xyz = evaluate_vector_description(polarization)
        logging.info(f'Evaluated {polarization=} to {eps_xyz=}. Should be {res_xyz=}')
        self._check_difference(eps_xyz, res_xyz)

    def _check_sph(self, polarization: str, res_sph: np.ndarray):
        """Check the evaluation in spherical coordinates. """
        
        eps_xyz = evaluate_vector_description(polarization)
        eps_sph = cart_to_sph(eps_xyz)
        self._check_difference(eps_sph, res_sph)

    def _check_conversion(self, eps_xyz):
        """ Check conversion between cartesian and spherical coordinates in both ways. """

        eps_sph = cart_to_sph(eps_xyz)
        eps_xyz_2 = sph_to_cart(eps_sph)
        self._check_difference(eps_xyz, eps_xyz_2)

    # Test cases
    def test_linear_polarization(self):
        """Test linear polarizations. """

        polarizations = ['x', 'y', 'z']
        results_xyz = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        results_sph = [np.array([1, 0, -1]) / np.sqrt(2), np.array([1j, 0j, 1j]) / np.sqrt(2), np.array([0, 1, 0])]

        # Check the conversion to spherical coordinates is correct
        for res_xyz in results_xyz:
            self._check_conversion(res_xyz)

        # Check that my definition of the spherical vectors is correct (this test is not strictly necessary)
        for (res_xyz, res_sph) in zip(results_xyz, results_sph):
            self._check_difference(cart_to_sph(res_xyz), res_sph)

        # Compare the results in cartesian coordinates
        for pol, res_xyz in zip(polarizations, results_xyz):
            self._check_cart(pol, res_xyz)

        # Compare the results in spherical coordinates
        for pol, res_sph in zip(polarizations, results_sph):
            self._check_sph(pol, res_sph)

    def test_rcp_polarization(self):
        """Test right circular polarization. """

        # Different variations of RCP
        polarizations = ['x + iy', '(x + i*y)', '1/sqrt(2) * (x + iy)', '1/sqrt(2)(x + iy)', '(x + iy)/sqrt(2)']
        res_xyz = np.array([1, 1j, 0]) / np.sqrt(2)
        res_sph = np.array([0, 0, -1])

        # Check the conversion to spherical coordinates is correct
        self._check_conversion(res_xyz)

        # Check that my definition of the spherical vectors is correct (this test is not strictly necessary)
        self._check_difference(cart_to_sph(res_xyz), res_sph)

        # Compare the results in cartesian coordinates
        for pol in polarizations:
            self._check_cart(pol, res_xyz)

        # Compare the results in spherical coordinates
        for pol in polarizations:
            self._check_sph(pol, res_sph)

    def test_lcp_polarization(self):
        """Test left circular polarization. """

        # Different variations of RCP
        polarizations = ['x - iy']
        res_xyz = np.array([1, -1j, 0]) / np.sqrt(2)
        res_sph = np.array([1, 0, 0])

        # Check the conversion to spherical coordinates is correct
        self._check_conversion(res_xyz)

        # Check that my definition of the spherical vectors is correct (this test is not strictly necessary)
        self._check_difference(cart_to_sph(res_xyz), res_sph)

        # Compare the results in cartesian coordinates
        for pol in polarizations:
            self._check_cart(pol, res_xyz)

        # Compare the results in spherical coordinates
        for pol in polarizations:
            self._check_sph(pol, res_sph)

    def test_long_string(self):
        """Evaluate some weird long polarization string. """

        pol = 'ix*sqrt(1) + 3x - 5(y + 3z)sin(pi/4) + z/sqrt(5)'

        # Get the corresponding polarization
        res_xyz = np.array(
            [1j + 3, 
            -5 * np.sin(np.pi / 4),
            -15 * np.sin(np.pi / 4) + 1 / np.sqrt(5)
            ]
        )
        # Normalize it
        res_xyz = res_xyz / np.sqrt(np.sum(res_xyz * res_xyz.conj()))

        self._check_cart(pol, res_xyz)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(filename='test_PolarizationUtil.log', level=logging.INFO)
    # Run the tests
    unittest.main()