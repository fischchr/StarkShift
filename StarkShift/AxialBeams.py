import numpy as np
from .BottleBeamUtils import I_bob
from .GaussianBeamUtils import *
from scipy.interpolate import RegularGridInterpolator
from pint import UnitRegistry


class AxialBeam:
    """Class for evaluating axially symmetric beams. """
    def __init__(self, I0: float, k: np.ndarray, r0: np.ndarray, ureg: UnitRegistry):
        """Constructor
        
        # Arguments
        * I0::float - Intensity scaling factor.
        * k::np.array(3) - Wave vector of the beam. 
        * r0::np.array(3) - Position of the waist.
        * ureg::UnitRegistry - pint.UnitRegistry object.
        """
        
        self._k = k
        self._r0 = r0
        self._u = ureg
        self._I0 = I0
    
    @property
    def k(self):
        """Get the k-vector of the beam. """

        return self._k

    @k.setter
    def k(self, k):
        """Set the k-vector of the beam. """

        self._k = k
    
    @property
    def k_hat(self):
        """Get the propagation direction of the beam. """

        if hasattr(self.k, 'to'):
            k_unitless = self.k.magnitude
            return k_unitless / np.sqrt(np.sum(k_unitless**2))
        else:
            return self.k / np.sqrt(np.sum(self.k**2))
    
    @property
    def k_mag(self):
        """Get the the magnitude of the k vector |k|. """

        return (np.sqrt(np.sum(self.k**2))).to('1/m')
        
    @property
    def lam(self):
        """Get the wavelength of the beam. """

        return (2*np.pi / self.k_mag).to('nm')
    
    @property
    def omega(self):
        """Get the angular frequency omega of the beam. """

        return (self.k_mag * self.u('c')).to('Hz')

    @property
    def nu(self):
        """Get the frequency nu of the beam. """

        return self.omega / (2 * np.pi)
        
    @property
    def r0(self):
        """Get the location of the waist. """

        return self._r0

    @r0.setter
    def r0(self, r0):
        """Set the location of the waist. """

        self._r0 = r0
    
    @property
    def I0(self):
        """Get the intensity scaling factor I0. """

        return self._I0
    
    @I0.setter
    def I0(self, I0):
        """Set the intensity scaling factor I0. """

        self._I0 = I0
        
    @property
    def units(self):
        """Get the unit registry. """

        return self._u
    
    @property
    def u(self):
        """Shorthand for self.units. """

        return self.units
    
    def eval_beam_coordinates(self, z_ax: np.array, r_perp: np.array) -> np.array:
        """Evaluate the intensity of the beam in the cooordinate system defined by the beam.
        
        # Arguments
        * z_ax::np.array(N) - Vector of axial positions.
        * r_perp::np.array(N) - Vector of radial positions.
        
        # Returns
        * I(z_ax, r_perp)::np.array(N)
        """
        
        assert z_ax.shape == r_perp.shape
        
        return self.I0 * np.ones(z_ax.shape)
    
    def I(self, x: np.array, y: np.array, z: np.array) -> np.array:
        """Evaluate the intensity of the beam in cartesian coordinates.
        
        # Arguments
        * x::np.array(N) - x coordinates of N points.
        * y::np.array(N) - y coordinates of N points.
        * z::np.array(N) - z coordinates of N points.
        
        # Returns
        * I(z_ax, r_perp)::np.array(N)
        """

        # Make sure x, y, and z are vectors. This way also an intput I(0, 0, 0) works.
        x = self._vectorize(x)
        y = self._vectorize(y)
        z = self._vectorize(z)

        z_ax, r_perp = self.convert_xyz_to_beam_coordinates(x, y, z)
        return self.eval_beam_coordinates(z_ax, r_perp)
    
    def eval_sph(self, r: np.array, theta: np.array, phi: np.array) -> np.array:
        """Evaluate the intensity of the beam in spherical coordinates.
        
        # Arguments
        * r::np.array(N) - r coordinates of N points.
        * theta::np.array(N) - theta coordinates of N points.
        * phi::np.array(N) - phi coordinates of N points.
        
        # Returns
        * I(z_ax, r_perp)::np.array(N)
        """
        
        z_ax, r_perp = self.convert_sph_to_beam_coordinates(r, theta, phi)
        return self.eval_beam_coordinates(z_ax, r_perp)
    
    def eval_xy_meshgrid(self, x: np.array, y: np.array, z0: float) -> np.array:
        """Evaluate the intensity in the xy plane at z0. 
        
        # Arguments
        * x::np.array(Nx) - x coordinates.
        * y::np.array(Ny) - y coordinates.
        * z0::float - z position
        
        # Returns
        * I(x, y)::np.array(Nx, Ny)
        """
        
        xx, yy = np.meshgrid(x.to('m').magnitude, y.to('m').magnitude, indexing='ij') * self.u('m')
        
        s = xx.shape
        
        # Cast it into vector
        xx_v = np.reshape(xx, (xx.size, ))
        yy_v = np.reshape(yy, (yy.size, ))
        zz_v = z0 * np.ones(xx_v.shape)
        
        # Evaluate
        I_v = self.I(xx_v, yy_v, zz_v)

        # Cast back to matrix form
        I = np.reshape(I_v, s)
        
        # Cast to vector form if x or y was a scalar
        if np.min(s) == 1:
            I = np.reshape(I, (I.size,))
            
        return I
    
    def eval_xz_meshgrid(self, x: np.array, y0: float, z: np.array) -> np.array:
        """Evaluate the intensity in the xz plane at y0. 
        
        # Arguments
        * x::np.array(Nx) - x coordinates.
        * y0::float - y postion.
        * z::np.array(Nz) - z coordinates.
        
        # Returns
        * I(x, z)::np.array(Nx, Nz).
        """
        
        xx, zz = np.meshgrid(x.to('m').magnitude, z.to('m').magnitude, indexing='ij') * self.u('m')
        
        s = xx.shape
        
        # Cast it into vector
        xx_v = np.reshape(xx, (xx.size, ))
        yy_v = y0 * np.ones(xx_v.shape)
        zz_v = np.reshape(zz, (zz.size, ))
  
        # Evaluate
        I_v = self.I(xx_v, yy_v, zz_v)

        # Cast back to matrix form
        I = np.reshape(I_v, s)
        
        # Cast to vector form if x or y was a scalar
        if np.min(s) == 1:
            I = np.reshape(I, (I.size,))
            
        return I
                  
    def convert_xyz_to_beam_coordinates(self, x: np.array, y: np.array, z: np.array) -> tuple:
        """Convert the cartesian coordinates x, y, z into beam coordinates z_ax, r_perp.

        This function performs a coordinate transformation from cartesian coordinates into the 
        cylindrically symmetric coordinate system defined by the beam. For a vector 

            r = (x, y, z)

        we first have to shift to a cartesian coordinate system relative to the beam waist, i.e.

            d = r - r0

        where r0 is pointing from the origin to the position of the waist.
        The axial position z_ax is given by the dot product

            z_ax = d.k_hat

        where k_hat = k / |k| is the direction of the beam.
        The radial position r_perp can then be calculated using the Pythagorean equation 

            r_perp = sqrt(|d|^2 - z_ax^2) = sqrt(d.d - z_ax^2)

        This transformation can be performed on N points simultaneously. We define a (3, N) matrix

            r_m[:, j] = (x_j, y_j, z_j)

        containing the coordinates of all N points as columns. We then define a matrix 

            k_m[:, j] = k_hat

        which conatins the vector k_hat N times. Now we can use the function np.einsum to perform
        the dot product column-wise, i.e.

            np.einsum('ij,ij->j', r_m, k_m)

        results in a vector of length N where element j is the dot product r_j.k_hat.

        # Arguments
        * x::np.array(N) - x coordinates of N points.
        * y::np.array(N) - y coordinates of N points.
        * z::np.array(N) - z coordinates of N points.

        # Returns
        (z_ax_v::np.array(N), z_perp_v::np.array(N)) - Axial and radial position along the beam.
        """

        # Check arguments
        assert isinstance(x.magnitude, np.ndarray) and \
               isinstance(y.magnitude, np.ndarray) and \
               isinstance(z.magnitude, np.ndarray), 'x, y and z must be the same type.'
        assert x.shape == y.shape and y.shape == z.shape, 'x, y and z must have the same dimensions.'

        # Get number of points
        N = x.shape[0]

        # Make r vector matrix r_m with r_m[:, j] = r_j = (x, y, z)_j
        r_m = np.stack([x, y, z], axis=0)
        #assert r_m.shape == (3, N)

        # Calculate the matrix of relative position vectors d_m with d_m[:, j] = d_j = r_j - self._z0
        # where d_j is the vector pointing from the beam waist to the point r_j = (x, y, z)_j
        d_m = r_m - self.r0[:, None]
        #assert d_m.shape == (3, N)

        # Calculate the latteral position z_ax_v of each point along the beam.
        # The latteral position is given by the dot product of the k/|k| = self.k_hat vector of the beam 
        # and the position of the point relative to the beam waist (d_j).
        # Here z_ax_v[j] = np.dot(self.k_hat, d_j)
        z_ax_v = np.einsum('ij,ij->j', self.k_hat[:, None], d_m) 
        #assert z_ax_v.shape == (N,)

        # Calculate the radial distance r_perp_v of each point to the beam.
        # The radial distance is given by 
        # r_perp_v[j] = sqrt(|d_j|^2 - z_v^2)
        r_perp_v = np.sqrt(np.einsum('ij,ij->j', d_m, d_m) - z_ax_v**2)
        #assert r_perp_v.shape == (N,)

        return z_ax_v, r_perp_v
        
    def convert_sph_to_beam_coordinates(self, r: np.array, theta: np.array, phi: np.array) -> tuple:
        """Convert the spherical coordinates r, theta, phi into beam coordinates z_ax, r_perp. """
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return self.convert_xyz_to_beam_coordinates(x, y, z)
    
    def _vectorize(self, x) -> np.array:
        """Cast x into an array if x is not yet an array. """

        if not isinstance(x.magnitude, np.ndarray):
            x = np.array([x.to('m').magnitude]) * self.u('m')

        return x
    

class GaussianBeam(AxialBeam):
    """Implementation of a Gaussian beam. """

    def __init__(self, P: float, k: np.ndarray, w0: float, r0: np.ndarray, ureg: UnitRegistry):
        """Constructor

        # Arguments
        * P::float - Total beam power.
        * k::np.array(3) - k-vector of the beam. lam = 2 pi / |k|
        * w0::float - Waist of the beam.
        * r0::np.array(3) - Location of the beam center.
        * ureg::UnitRegistry - pint.UnitRegistry object.
        """

        self._w0 = w0 
        self._P = P
        self._k = k
        self._r0 = r0
        self._u = ureg


    # Beam waist
    @property
    def w0(self):
        """Get the waist of the beam. """

        return self._w0

    @w0.setter
    def w0(self, w0):
        """Set the waist of the beam. """

        self._w0 = w0

    # Rayleigh range
    @property
    def zR(self):
        return get_zR(self.w0, self.lam)

    # Beam power
    @property
    def P(self):
        """Get the total power of the beam. """

        return self._P

    @P.setter
    def P(self, P):
        """Set the total power of the beam. """

        self._P = P
    
    # Beam intensity
    @property
    def I0(self):
        """Get the intensity of the beam at the waist. """
        return get_I0(self.P, self.w0)

    @I0.setter
    def I0(self, I0):
        """Set the intensity of the beam at the waist. """
        self.P = get_P0(I0, self.w0)

    
    def eval_beam_coordinates(self, z_ax: np.array, r_perp: np.array) -> np.array:
        """Evaluate the intensity of the beam in the cooordinate system defined by the beam.
        
        # Arguments
        * z_ax::np.array(N) - Vector of axial positions.
        * r_perp::np.array(N) - Vector of radial positions.
        
        # Returns
        * I(z_ax, r_perp)::np.array(N)
        """
        
        assert z_ax.shape == r_perp.shape

        # Calculate the beam radius for each point
        w = get_w(self.w0, self.lam, z_ax)
        
        # Calculate the intensity for each point
        I = self.I0 * (self.w0 / w)**2 * np.exp(-2 * (r_perp / w)**2)
        
        return I


class BottleBeam(AxialBeam):
    """Bottle beam. """
    def __init__(self, I0: float, k: np.ndarray, a: float, NA: float, r0: np.ndarray, ureg: UnitRegistry, d_max: float, 
                 N_eval: int = 51):
        """Constructor

        # Arguments
        * I0::float - Intensity scaling factor.
        * k::np.array(3) - k-vector of the beam. lam = 2 pi / |k|
        * a::float - Scaled radius of the phase-shifted region.
        * NA::float - Numerical apperture.
        * r0::np.array(3) - Location of the beam center.
        * ureg::UnitRegistry - pint.UnitRegistry object.
        * d_max::float - Maximum distance to which the beam can be evaluated at.
        * N_eval::int - Number of grid points on which the intensity of the bottle beam is exactly evaluated.
        """

        # Initialize the axial beam
        super(BottleBeam, self).__init__(I0, k, r0, ureg)
        
        # Set beam properties
        self._a = a
        self._NA = NA
        self._I0 = I0
        
        # Set interpolation
        self._N_eval = N_eval

        if np.abs(d_max.to('um').magnitude) < 1e-6:
            raise ValueError('d_max must be larger than 0.')
        self._rho_max = self.rho(d_max)
        self._mu_max = self.mu(d_max)
         
        # Interpolate the beam 
        self.interpolate_beam()
               
    @property
    def a(self):
        """Get the scaled radius of the phase-shifted region. """

        return self._a
    
    @a.setter
    def a(self, a):
        """Get the scaled radius of the phase-shifted region. Triggers new interpolation of the beam. """

        self._a = a
        self.interpolate_beam()
        
    @property
    def NA(self):
        """Get the numerical aperture of the imaging system. """

        return self._NA
    
    @NA.setter
    def NA(self, NA):
        """Set the numerical aperture of the imaging system. Triggers new interpolation of the beam. """

        self._NA = NA
        self.interpolate_beam()  
        
    def rho(self, r: float):
        """Get the scaled radius rho. """

        return (2*np.pi / self.lam * self.NA * r).to_base_units().magnitude
    
    def mu(self, z):
        """Get the scaled axial position mu. """

        return (2*np.pi / self.lam * self.NA**2 * z).to_base_units().magnitude
    
    def interpolate_beam(self):   
        """Interpolate the bottle beam. """

        # Evaluate the beam in rho = [-20, 20]
        R_range = (self._rho_max / (2 * np.pi / self.lam * self.NA)).to('um').magnitude
        Rs = np.linspace(-R_range, R_range, self._N_eval) * self.u('um')
        
        # Evaluate the beam in mu = [-50, 50]
        Z_range = (self._mu_max / (2 * np.pi / self.lam * self.NA**2)).to('um').magnitude
        Zs = np.linspace(-Z_range, Z_range, self._N_eval) * self.u('um')
        
        # Evaluate the beam intensity at each point
        self.I_exact = np.zeros((self._N_eval, self._N_eval))
        
        for i, R in enumerate(Rs):
            rho = self.rho(R)
            for j, Z in enumerate(Zs):
                mu = self.mu(Z)
                self.I_exact[i, j] = I_bob(rho, mu, self.a) 
        
        # Interpolate the exact intensity on a meshgrid
        Rs_um = Rs.to('um').magnitude
        Zs_um = Zs.to('um').magnitude
                
        self._I_interpolation = RegularGridInterpolator((Rs_um, Zs_um), self.I_exact, 
                                                        fill_value=0, bounds_error=False)
        
    def eval_beam_coordinates(self, z_ax, r_perp):
        """Evaluate the intensity of the beam in the cooordinate system defined by the beam.
        Overwrites the AxialBeam method.
        
        # Arguments
        * z_ax::np.array(N) - Vector of axial positions.
        * r_perp::np.array(N) - Vector of radial positions.
        
        # Returns
        * I(z_ax, r_perp)::np.array(N)
        """
        
        # Convert distances to micrometer
        z_ax_um = z_ax.to('um').magnitude
        r_perp_um = r_perp.to('um').magnitude
                
        # Stack the vectors
        x = np.stack((r_perp_um, z_ax_um), axis=1)
        
        return self.I0 * self._I_interpolation(x)

    @property
    def I0(self):
        return self.eval_beam_coordinates(np.array([0]), np.array([0]))[0]


if __name__ == "__main__":
    pass 

    from matplotlib import pyplot as plt
    import pint

    u = pint.UnitRegistry()

    # Test the Gaussian beam class
    lam = 532 * u('nm')
    k = 2*np.pi * np.array([0, 0, 1]) / lam
    r0 = np.array([0, 0, 0]) * u('um')
    P = 10 * u('mW')
    w0 = 10 * u('um')

    beam = GaussianBeam(P, k, w0, r0, u)

    # Evaluate transverse to the beam in 1d
    r = np.linspace(-w0.to('um').magnitude, w0.to('um').magnitude, 102) * u('um')
    z = 0 * u('um')

    z_v = z * np.ones(r.shape)

    plt.plot(r.magnitude, beam.eval_beam_coordinates(z_v, r).magnitude)
    plt.xlabel('r (um)')
    plt.show()

    # Evaluate along the beam in 1d
    r = 0 * u('um')
    zR = beam.zR.to('um').magnitude
    z_v = np.linspace(-zR, zR, 102) * u('um')

    r_v = r * np.ones(z_v.shape)

    plt.plot(z_v.magnitude, beam.eval_beam_coordinates(z_v, r_v).magnitude)
    plt.xlabel('z (um)')
    plt.show()

    # Test the bottle beam class
    I0 = 1 * u('W/cm^2')
    lam = 532 * u('nm')
    k = 2*np.pi * np.array([0, 0, 1]) / lam
    r0 = np.array([1, 0, 1]) * u('um')

    a = 0.62
    NA = 0.4

    bob = BottleBeam(I0, k, a, NA, r0, u, N_eval=51)

    # Evaluate along the beam in 1d
    r = np.linspace(-1, 1, 102) * u('um')
    z = 0 * u('um')

    z_v = z * np.ones(r.shape)

    plt.plot(r.magnitude, bob.eval_beam_coordinates(z_v, r).magnitude)
    plt.xlabel('r (um)')
    plt.show()

    # Evaluate the beam in the xy plane
    x = np.linspace(-2, 2, 51) * u('um')
    y = np.linspace(-1, 1, 51) * u('um')
    z = 0 * u('um')
    I = bob.eval_xy_meshgrid(x, y, z)
    lim = np.max(I.magnitude)
    plt.contourf(y.magnitude, x.magnitude, I.magnitude, cmap='coolwarm', vmin=-lim, vmax=lim,levels=10)
    plt.xlabel('y (um)')
    plt.ylabel('x (um)')
    plt.show()

    # Evaluate the beam in the xz plane
    x = np.linspace(-2, 2, 52) * u('um')
    y = 0 * u('um')
    z = np.linspace(-10, 10, 51) * u('um')

    I = bob.eval_xz_meshgrid(x, y, z)
    lim = np.max(I.magnitude)
    plt.contourf(z.magnitude, x.magnitude, I.magnitude, cmap='coolwarm', vmin=-lim, vmax=lim,levels=10)
    plt.xlabel('z (um)')
    plt.ylabel('x (um)')
    plt.show()