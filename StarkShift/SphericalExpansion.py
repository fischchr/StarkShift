
from .SphericalExpansionFunctions import expand_complex
import numpy as np
from .AxialBeams import AxialBeam


class SphericalExpansion:
    """Class for expanding a function f(r, theta, phi) in spherical harmonics. """

    def __init__(self, N_r: int, N_theta: int, L_max: int):
        """Constructor. 
        
        # Arguments
        * N_r::int - Number of radial grid points.
        * N_theta::int - Number of grid points (altitude). The number of azimuthal grid points N_phi = 2 * N_theta
        * L_max::int - Order of expansion in spherical harmonics Y_{l,m}.
        """

        # Define grid for fitting with spherical harmonics
        self.r_v = np.zeros(N_r)
        self.N_theta = N_theta
        self.L_max = L_max
        # Allocate memory for the expansion coefficients
        self.coeffs = np.zeros([N_r, 2, L_max + 1, L_max + 1], dtype=np.complex)
        # Allocate memory for the function that was interpolated
        self._f = None

    def interpolate(self, f, r_i: float, r_o: float):
        """Interpolate a function f. 
        
        # Arguments
        * f::function - Function f(r, theta, phi) to interpolate.
        * r_i::float - Inner interpolation limit.
        * r_o::float - Outer interpolation limit.
        """

        # Store the input
        self._f = f
        self._range = [r_i, r_o]

        # Get the size
        N_r, _, _, _ = self.coeffs.shape
        L_max = self.L_max
        N_theta = self.N_theta

    
        # Define a vector of radial points
        self.r_v = np.linspace(r_i, r_o, N_r)

        # Interpolate f for each point in r_v
        for i, r in enumerate(self.r_v):
            def f_ang(tt: np.array, pp: np.array):
                """Function for evaluating f(r=r_v[i], t, p). """

                # Convert the meshgrid tt and pp into vectors
                t_v = tt.reshape(2 * N_theta**2)
                p_v = pp.reshape(2 * N_theta**2)
                r_v = r * np.ones(p_v.shape)

                # Evaluate f for all points tt and pp for fixed r_v[i]
                f_v = f(r_v, t_v, p_v)

                # Convert f back to matrix form
                f_m = f_v.reshape(N_theta, 2 * N_theta)

                return f_m
        
            # Calculate the spherical expansion of f for r_v[i] and store them
            self.coeffs[i, :, :, :] = expand_complex(f_ang, N_theta, L_max)

    def get_coeff(self, r: float, k: int, q: int) -> np.complex:
        """Get the expansion coefficient f_{k,q}(r). 
        
        # Arguments
        * r::float - Point at which the expansion coefficient f_{k,q} is evaluated.
        * k::int - Rank of the expansion coefficient.
        * q::int - Element of the expansion coefficient.
        """


        # Check that a function has been interpolated
        assert not (self._f is None), 'No function has been interpolated.'

        # Get the coefficients f_{k, q}(r_v)
        if q < 0:
            fkq = self.coeffs[:, 1, k, -q]
        else:
            fkq = self.coeffs[:, 0, k, q]

        # Interpolate f_{k, q}(r)
        res = np.interp(r, self.r_v, fkq)

        # Put in the normalization we want to use
        return np.sqrt((2*k + 1) / (4*np.pi)) * res


class SphericalBeamExpansion(SphericalExpansion):
    """"Implementation of SphericalExpansion which interpolates an axial beam and deals with units. """
    
    def __init__(self, beam: AxialBeam, N_r: int, N_theta: int, L_max: int, r_i: float, r_o: float):
        """Constructor. 
        
        # Arguments
        * beam::AxialBeam - Beam object which is expanded.
        * N_r::int - Number of radial grid points.
        * N_theta::int - Number of grid points (altitude). The number of azimuthal grid points N_phi = 2 * N_theta
        * L_max::int - Order of expansion in spherical harmonics Y_{l,m}.
        * r_i::float - Inner interpolation limit in SI units.
        * r_o::float - Outer interpolation limit in SI units.
        """
        super(SphericalBeamExpansion, self).__init__(N_r, N_theta, L_max)

        self.beam = beam
        self.u = self.beam.units
        
        # Interpolate the beam
        self.interpolate(r_i, r_o)


    @property
    def units(self):
        """Get the unit registry. """

        return self.u


    def interpolate(self, r_i: float, r_o: float):
        """Wrapper for the interpolation function that deals with the units and interpolates `self.beam`. 
        
        # Arguments
        * r_i::float - Inner interpolation limit.
        * r_o::float - Outer interpolation limit.
        """

        # Remove the units from the intensity of the beam
        def f(r, tt, pp):
            # Put the correct unit on r
            r = r * self.u('a_u_length')
            # Evaluate the intensity
            I = self.beam.eval_sph(r, tt, pp)
            # Return the intensity in atomic units
            return I.to('a_u_energy/(a_u_time * a_u_length^2)').magnitude

        # Remove the units from the end points
        r_i_au = r_i.to('a_u_length').magnitude
        r_o_au = r_o.to('a_u_length').magnitude

        # Use the interpolation function of the parent class
        super(SphericalBeamExpansion, self).interpolate(f, r_i_au, r_o_au)

    def get_coeff(self, r: float, k: int, q: int):
        """Wrapper for getting the expansion coefficients f_{k,q}(r) that puts the correct unit back on. 

        # Arguments
        * r::float - Point at which the expansion coefficient f_{k,q} is evaluated in SI units.
        * k::int - Rank of the expansion coefficient.
        * q::int - Element of the expansion coefficient.
        """

        # Convert r to atomic units
        r_au = r.to('a_u_length').magnitude
        # Get the coefficient without unit.
        c = super(SphericalBeamExpansion, self).get_coeff(r_au, k, q)

        # Put the unit of intensity back on the coefficient.
        c = c * self.u('a_u_energy/(a_u_time * a_u_length^2)')

        return c