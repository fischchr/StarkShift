import logging
import unittest
import numpy as np

# Import the functions to test
from StarkShift.PolarizationUtil import cart_to_sph, sph_to_cart
from StarkShift.PolarizationUtil import evaluate_vector_description
from StarkShift.PolarizationUtil import transform_polarization, get_polarization_vector

# Set up logging
logging.basicConfig(filename='tests/logs/test_PolarizationUtil.log', level=logging.INFO)


# Define the testcase
class TestPolarizationUtil(unittest.TestCase):
    """Test the conversion of strings to polarization vectors. """

    ### Helper functions ###
    def _check_difference(self, vec_1: np.ndarray, vec_2: np.ndarray, eps: float = 1e-12):
        """Check that the difference between two vectors is small. """

        # Calculate the difference
        diff = vec_1 - vec_2
        # Calculate the magntiude
        mag = np.sum(diff * diff.conj())
        # Check that the magnitude is small
        self.assertTrue(mag < eps)

    # For test 1
    def _check_cart(self, polarization: str, res_xyz: np.ndarray):
        """Check the evaluation in caresian coordinates. 
        Used for testing `evaluate_vector_description`.

        # Arguments
        * polarization::str - Description of the polarization in cartesian coordinates.
        * res_xyz::vector(3) - Expected polarization vector in cartesian coordinates.
        """

        eps_xyz = evaluate_vector_description(polarization)
        logging.info(f'_check_cart evaluated {polarization=} to {eps_xyz=}. Should be {res_xyz=}')
        self._check_difference(eps_xyz, res_xyz)

    def _check_sph(self, polarization: str, res_sph: np.ndarray):
        """Check the evaluation in spherical coordinates. 
        Used for testing the combination of `evaluate_vector_description` and `cart_to_sph`.

        # Arguments
        * polarization::str - Description of the polarization in cartesian coordinates.
        * res_sph::vector(3) - Expected polarization vector in spherical coordinates defined by e_z = [0, 0, 1].
        """
        
        eps_xyz = evaluate_vector_description(polarization)
        eps_sph = cart_to_sph(eps_xyz)
        self._check_difference(eps_sph, res_sph)

    def _check_conversion(self, eps_xyz):
        """ Check conversion between cartesian and spherical coordinates in both ways. 
        Used for testing `cart_to_sph` and `sph_to_cart`.

        # Arguments
        * eps_xyz::vector(3) - Polarization vector in cartesian coordinates.
        """

        eps_sph = cart_to_sph(eps_xyz)
        eps_xyz_2 = sph_to_cart(eps_sph)
        self._check_difference(eps_xyz, eps_xyz_2)

    # For test 2
    def _check_transformed_vectors(self, e_p: np.ndarray, e_q: np.ndarray, res: np.ndarray):
        """Transform polarization vectors and check the result. 
        Used for testing `transform_polarization`. 

        # Arguments
        * e_p::vector(3) - Polarization vector in cartesian coordinates.
        * e_q::vector(3) - Quantizatoin axis in cartesian coordinates.
        * res::vector(3) - The polarization vector in the cartesian coordinate system in which e_q corresponds to e_z.
        """

        # Transform the polarization vector
        e_p_prime = transform_polarization(e_p, e_q)
        
        # Log the result
        logging.info(f'_check_transformed_vectors evaluated {e_p=} to {e_p_prime=} ({e_p=}). Should be {res=}')

        # Check the result
        self._check_difference(e_p_prime, res)

    # For test 3
    def _check_full_stack(self, epsilon: str, e_q: np.ndarray, res: np.ndarray):
        """Evaluate a polarization string, transform it to the coordinate system defined by e_q and check the result. 
        Used for testing `get_polarization_vector`.

        # Arguments
        * epsilon::str - Description of the polarization in cartesian coordinates.
        * e_q::vector(3) - Quantization axis in cartesian coordinates.
        * res::vector(3) - Expected polarization vector in spherical coordinates defined by e_q
        """

        # Transform the string describing the polarization to the coordinate system defined by e_q
        e_q_prime_sph = get_polarization_vector(epsilon, e_q)

        # Log the result
        logging.info(f'_check_full_stack: evaluated {epsilon=} to {e_q_prime_sph=} ({e_q=}). Should be {res=}')

        # Check the result
        self._check_difference(e_q_prime_sph, res)

    ### Test cases ###
    # 1. Test parsing of strings and conversion between cartesian and spherical coordinates
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

    # 2. Test the transformation of a polarization vector from cartesian coordinates to the coordinate system defined by a given quantization axis
    def test_transform_polarization_ez(self):
        """Test the function `transform_polarization` when the quantization axis is e_z. """

        # Quantization axis along z should not change e_p
        e_q = np.array([0, 0, 1])

        # Polarizations we test
        e_ps = [
            evaluate_vector_description('x'), 
            evaluate_vector_description('y'), 
            evaluate_vector_description('z'), 
            evaluate_vector_description('x+iy'), 
            evaluate_vector_description('x-iy')]

        # Run the tests
        for e_p in e_ps:
            # e_p_prime should be equal to e_p
            res = e_p
            self._check_transformed_vectors(e_p, e_q, res)

    def test_transform_polarization_ex(self):
        """Test the function `transform_polarization` when the quantization axis is e_x. """

        # Quantization axis along x
        e_q = np.array([1, 0, 0])

        # Polarizations we test
        e_ps = [
            evaluate_vector_description('x'), # e_x should become e_z
            evaluate_vector_description('y'), # e_y should stay e_y
            evaluate_vector_description('z'), # e_z should become -e_x
            evaluate_vector_description('x+iy'), # RCP should become z+iy
            evaluate_vector_description('x-iy') # LCP should become z-iy
        ]

        # Results we expect
        results = [
            evaluate_vector_description('z'),
            evaluate_vector_description('y'),
            evaluate_vector_description('-x'),
            evaluate_vector_description('z+iy'),
            evaluate_vector_description('z-iy')
        ]

        # Run the tests
        for e_p, res in zip(e_ps, results):
            self._check_transformed_vectors(e_p, e_q, res)

    def test_transform_polarization_ey(self):
        """Test the function `transform_polarization` when the quantization axis is e_y. """

        # Quantization axis along y
        e_q = np.array([0, 1, 0])

        # Polarizations we test
        e_ps = [
            evaluate_vector_description('x'), # e_x should become -e_y
            evaluate_vector_description('y'), # e_y should become e_z
            evaluate_vector_description('z'), # e_z should become -e_x
            evaluate_vector_description('x+iy'), # RCP should become -y+iz
            evaluate_vector_description('x-iy') # LCP should become -y-iz
        ]

        # Results we expect
        results = [
            evaluate_vector_description('-y'),
            evaluate_vector_description('z'),
            evaluate_vector_description('-x'),
            evaluate_vector_description('-y+iz'),
            evaluate_vector_description('-y-iz')
        ]

        # Run the tests
        for e_p, res in zip(e_ps, results):
            self._check_transformed_vectors(e_p, e_q, res)

    # 3. Test the full transformation of a string to a spherical vector in the coordinate system defined by a given quantization axis
    def test_get_polarization_vector_ez(self):
        """Test the function `get_polarization_vector` when the quantization axis is e_z. """

        # Quantization axis along z
        e_q = np.array([0, 0, 1])

        # Define the polarizations we test
        epsilons = [
            'x', # remains x
            'y', # remains y
            'z', # remains z
            'x+iy', # remains rcp
            'x-iy', # remains lcp
        ]

        # Define the expected results in spherical coordinates
        results = [
            cart_to_sph(evaluate_vector_description('x')), 
            cart_to_sph(evaluate_vector_description('y')), 
            cart_to_sph(evaluate_vector_description('z')), 
            cart_to_sph(evaluate_vector_description('x+iy')), 
            cart_to_sph(evaluate_vector_description('x-iy')), 
        ]

        # Run the tests
        for eps, res_sph in zip(epsilons, results):
            self._check_full_stack(eps, e_q, res_sph)

    def test_get_polarization_vector_ex(self):
        """Test the function `get_polarization_vector` when the quantization axis is e_x. """

        # Quantization axis along x
        e_q = np.array([1, 0, 0])

        # Define the polarizations we test
        epsilons = [
            'x', # becomes z
            'y', # remains y
            'z', # becomes -x
            'x+iy', # becomes z+iy
            'x-iy', # becomes z-iy
        ]

        # Define the expected results in spherical coordinates
        results = [
            cart_to_sph(evaluate_vector_description('z')), 
            cart_to_sph(evaluate_vector_description('y')), 
            cart_to_sph(evaluate_vector_description('-x')), 
            cart_to_sph(evaluate_vector_description('z+iy')), 
            cart_to_sph(evaluate_vector_description('z-iy')), 
        ]

        # Run the tests
        for eps, res_sph in zip(epsilons, results):
            self._check_full_stack(eps, e_q, res_sph)

    def test_get_polarization_vector_ey(self):
        """Test the function `get_polarization_vector` when the quantization axis is e_y. """

        # Quantization axis along y
        e_q = np.array([0, 1, 0])

        # Define the polarizations we test
        epsilons = [
            'x', # becomes -y
            'y', # becomes z
            'z', # becomes -x
            'x+iy', # becomes -y+iz
            'x-iy', # becomes -y-iz
        ]

        # Define the expected results in spherical coordinates
        results = [
            cart_to_sph(evaluate_vector_description('-y')), 
            cart_to_sph(evaluate_vector_description('z')), 
            cart_to_sph(evaluate_vector_description('-x')), 
            cart_to_sph(evaluate_vector_description('-y+iz')), 
            cart_to_sph(evaluate_vector_description('-y-iz')), 
        ]

        # Run the tests
        for eps, res_sph in zip(epsilons, results):
            self._check_full_stack(eps, e_q, res_sph)


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(filename='test_PolarizationUtil.log', level=logging.INFO)
    # Run the tests
    unittest.main()