import types
import numpy as np
import pyshtools as pysh


def generate_meshgrid(N_theta: int, N_phi: int) -> tuple:
    """Gernerate a grid on the unit sphere. 
    
    # Arguments:
    * N_theta::int - Lateral number of grid points.
    * N_phi::int - Longitudinal number of grid points.
    
    # Returns
    (tt:np.array(N_theta, N_phi), pp::np.array(N_theta, N_phi)) - Meshgrid on the unit sphere.
    """
    
    # Generate the grid of points in θ and ϕ
    theta = np.linspace(0, np.pi, N_theta)
    phi = np.linspace(0, 2*np.pi, N_phi)
    
    # Generate a mesh grid with matrix indexing
    tt, pp = np.meshgrid(theta, phi, indexing='ij')
    
    assert tt.shape == pp.shape and pp.shape == (N_theta, N_phi)
    
    return tt, pp


def generate_sh_grid(f: types.FunctionType, N_theta: int) -> pysh.SHGrid:
    """Evaluate a function on a regularly spaced grid on the unit sphere.
    
    # Arguments
    * f::function(theta, phi) - Function to evaluate.
    * N_theta::int - Number of latteral grid points. The number of longitudinal grid points N_phi = 2*N_theta.
    
    # Returns 
    grid::pysh.SHGrid - Instance of SHGrid.
    """
    
    # We need twice as many points in phi than in theta
    N_phi = 2 * N_theta
    
    # Generate a meshgrid
    tt, pp = generate_meshgrid(N_theta, N_phi)
    
    # Evaluate f on all mesh points
    f_array = f(tt, pp)
    
    # Instanciate SHGrid from f_array
    grid = pysh.SHGrid.from_array(f_array, grid='DH')
    
    return grid


def expand_real(f: types.FunctionType, N_theta: int, L_max: int = 0):
    """Expand a function in real spherical harmonics Zlm on the unit sphere.
    
    # Arguments
    * f::function(theta, phi) - Function to evaluate. f **must** be real-valued.
    * N_theta::int - Number of latteral grid points. The number of longitudinal grid points N_phi = 2*N_theta.
    * L_max::int - Number of spherical harmonics to us. Default is L_max = N_theta/2 - 1.
    
    # Returns 
    coeffs::np.array(2, L_max + 1, L_max + 1) - Array containing the coefficients. The first index is the sign of m.
                                                coeffs[s, l, m] is the coefficient associated with Z_{l, (-1)**s * m}.
    """

    # Make sure f is real valued
    # If we wouldn't do that the expansion below seems to do some sort of complex expansion
    # I just don't quite understand what it is.
    def f_real(*args):
        """Wrapper for f that only takes the real part. """
        return np.real(f(*args))
    
    # Evaluate f on the unit sphere.
    grid = generate_sh_grid(f_real, N_theta)
    
    # Set L_max = N_theta/2 if no argument was passed.
    if L_max <= 0:
        L_max = grid.lmax
    
    # Calculate the expansion. The normalization we want to use is
    # https://shtools.oca.eu/shtools/public/real-spherical-harmonics.html#orthonormalized
    coeffs = grid.expand(normalization='ortho', lmax_calc=L_max, csphase=-1).coeffs
    
    assert coeffs.shape == (2, L_max + 1, L_max + 1)
    
    return coeffs


def real_to_complex(coeffs_real: np.array) -> np.array:
    """Calculate the complex expansion coefficients from a set of real ones.

    # Arguments
    * coeffs_real::np.array(2, L, M) - Array containing the real coefficients. The first index is the sign of m.
                                       coeffs[s, l, m] is the coefficient associated with the real spherical harmonic
                                       Z_{l, (-1)**s * m}.

    # Returns
    * coeffs_comples::np.array(2, L, M) - Array containing the real coefficients. The first index is the sign of m.
                                          coeffs[s, l, m] is the coefficient associated with the complex spherical harmonic
                                          Y_{l, (-1)**s * m}.
    """

    # Get the maximum value of L
    L_max = coeffs_real.shape[1] - 1
    
    # Make sure the input has the right shape
    assert coeffs_real.shape == (2, L_max + 1, L_max + 1)
    
    # Allocate memory for the result
    coeffs_complex = np.zeros(coeffs_real.shape, dtype=np.complex)
    
    # Iterate over all values of L and m
    for l in np.arange(0, L_max + 1, 1):
        for m in np.arange(-l, l + 1, 1):
            # Get c_{l, |m|}
            c_real_p = coeffs_real[0, l, np.abs(m)]
            # Get c_{l, -|m|}
            c_real_n = coeffs_real[1, l, np.abs(m)]

            # Calculate the complex coefficient
            if m < 0:
                coeff = 1 / np.sqrt(2) * ((-1)**np.abs(m) * c_real_p - 1j * c_real_n)
                coeffs_complex[1, l, np.abs(m)] = coeff
            elif m > 0:
                coeff = 1 / np.sqrt(2) * (c_real_p + 1j * (-1)**np.abs(m) * c_real_n)
                coeffs_complex[0, l, np.abs(m)] = coeff
            else:            
                coeffs_complex[0, l, 0] = coeffs_real[0, l, 0]
                
    return coeffs_complex


def expand_complex(f: types.FunctionType, N_theta: int, L_max: int = 0):
    """Expand a real valued function in complex spherical harmonics Ylm on the unit sphere.
    
    # Arguments
    * f::function(theta, phi) - Function to evaluate. f must be real.
    * N_theta::int - Number of lateral grid points. The number of longitudinal grid points N_phi = 2*N_theta.
    * L_max::int - Number of spherical harmonics to us. Default is L_max = N_theta/2 - 1.
    
    # Returns 
    coeffs::np.array(2, L_max + 1, L_max + 1) - Array containing the complex expansion coefficients. The first index is the sign of m.
                                                coeffs[s, l, m] is the coefficient associated with Y_{l, (-1)**s * m}.
    """

    # Calculate the real coefficients
    real_coeffs = expand_real(f, N_theta, L_max)
    
    # Convert them to complex
    complex_coeffs = real_to_complex(real_coeffs)
    
    return complex_coeffs

