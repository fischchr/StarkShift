import numpy as np
import logging
from scipy.special import jv
from scipy.optimize import fmin
import scipy.integrate as integrate
from scipy.interpolate import interp1d


# Helper functions for the BottleBeam class
def r_integral_factor(r, rho, mu):
    """Evaluate r * exp(-r**2) * jv(0, rho * r) * exp(-1j * mu * r**2 / 2) """

    return r * np.exp(-r**2) * jv(0, rho * r) * np.exp(-1j * mu * r**2 / 2) 

def r_integral(rho, mu, r_i, r_f):
    real_part = integrate.quad(lambda r: np.real(r_integral_factor(r, rho, mu)), r_i, r_f)[0]
    imag_part = integrate.quad(lambda r: np.imag(r_integral_factor(r, rho, mu)), r_i, r_f)[0]

    return real_part + 1j * imag_part

def E_bob(rho, mu, a):
    pre = 2 / (1 - np.exp(-1))
    return pre * r_integral(rho, mu, 0, 1) - 2 * pre * r_integral(rho, mu, 0, a)

def I_bob(rho, mu, a):
    """Get the Intensity of a bottle beam I(rho, mu). """

    return np.abs(E_bob(rho, mu, a))**2

def _find_minimum(x: np.ndarray, y: np.ndarray, verbose: bool = False):
    """Function for finding the minimum of a function f(x) = y. 
    Raises a ValueError if the x_min is outside the interpolation range.
    """

    # Interpolate the function
    f = interp1d(x, y, kind='cubic', fill_value=0)
    
    # Get the initial guess
    x0 = x[np.argmin(y)]
    # Find the minimum value of f(x) = -y
    res = fmin(f, x0, disp=verbose, full_output=True)
    
    xmin = res[0][0]
    ymin = res[1]
    
    return xmin, ymin


def _find_maximum(x: np.ndarray, y: np.ndarray, verbose: bool = False):
    """Function for finding the maximum of a function f(x) = y. 
    Raises a ValueError if the x_max is outside the interpolation range.
    """

    # Find the minimum value of f(x) = -y
    xmin, ymin = _find_minimum(x, -y)

    # Get the maximum value
    xmax = xmin
    ymax = -ymin
    
    return xmax, ymax


def _find_saddle_point_direction(beam, verbose: bool = False) -> float:
    """Function for finding the direction of the saddle point of a bottle beam intensity profile. 

    Since a bottle beam is symmetric around the axis defined by the propagation direction of the beam, 
    the saddle point can be found in the plane parallel to the k-vector of the beam.
    For example, for a beam propagating along z, the saddle point can be found in the xz-plane (or yz).

    In general, this plane is defined by the beam coordinates (z, r_perp) which are rewritten in 
    polar coordinates (r, phi). The saddle point can be found by first finding the maximum 
    I_max(phi) = max I(r, phi) for every angle phi and then finding the minimum value I_saddle = min I_max(phi).
    The corresponding angle phi_saddle for which I_saddle = I_max(phi_saddle) is then returned.
    
    # Arguments
    * beam::BottleBeam - The beam.
    * verbose::bool - Additional output.

    # Returns
    phi_saddle::float - Direction of the saddle point.
    """

    # Get the unit registry
    ureg = beam.units
    
    # Define the points on which to evaluate the intensity
    r = np.linspace(0 * ureg('m'), beam._dmax, 250)
    phis = np.linspace(0, np.pi / 2, 100)
    
    # Allocate memory for the result
    I_max = np.zeros(phis.shape) * ureg('mW/um^2')
    
    # Iterate over each angle
    for i in range(phis.size):
        phi = phis[i] * np.ones(r.shape)
        
        # Define the coordinates relative to the focal point of the beam
        z = r * np.cos(phi)
        rperp = r * np.sin(phi)
        
        # Evaluate the intensity for all r values
        I = beam.eval_beam_coordinates(z, rperp)
        
        # Find the maximum intensity for each angle phi
        I_max[i] = np.max(I)

    if verbose:
        # Plot the intensity maximum in r as a function of angle phi
        from matplotlib import pyplot as plt
        plt.plot(phis, I_max.to('mW/um^2').magnitude)
        plt.xlabel('$\phi$ (rad)')
        plt.ylabel(f'$I_{{max}}(\phi)$ (mW/\u00B5m$^2$)')
        
    # Find the direction of the saddle point
    try:
        phi_saddle, I_saddle = _find_minimum(phis, I_max.to('mW/um^2').magnitude)
        I_saddle *= ureg('mW/um^2')
    except ValueError:
        if verbose:
            plt.show()
        raise ValueError(f'Could not find saddle point. Increase {beam._dmax=}.')

    if verbose:
        # Annotate saddle point
        plt.plot(phi_saddle, I_saddle.to('mW/um^2').magnitude, 'o')
        plt.show()
    
    return phi_saddle


def _find_saddle_point_radius(beam, phi_saddle: float, verbose: bool = True) -> float:
    """Function for finding the distance from the focus of a bottle beam to the saddle point of the intensity profile. 
    
    # Arguments
    * beam::BottleBeam - The beam.
    * phi_saddle::float - Direction of the saddle point.
    * verbose::bool - Additional output.

    # Returns
    r_saddle::float - Distance of the saddle point relative to the focus of the beam.
    """

    # Get the unit registry
    ureg = beam.units
    
    # Define the points on which to evaluate the intensity
    rs = np.linspace(0 * ureg('m'), beam._dmax, 250)

    # Calculate the beam coordinates
    z = rs * np.cos(phi_saddle)
    rperp = rs * np.sin(phi_saddle)
        
    # Evaluate the intensity for all r values
    I = beam.eval_beam_coordinates(z, rperp)

    if verbose:
        # Plot the intensity maximum  as a function of r
        from matplotlib import pyplot as plt
        plt.plot(rs.to('um').magnitude, I.to('mW/um^2').magnitude)
        plt.xlabel('$r$ (\u00B5m)')
        plt.ylabel(f'$I_{{max}}(\phi)$ (mW/\u00B5m$^2$)')
    try:
        r_saddle, I_saddle = _find_maximum(rs.to('um').magnitude, I.to('mW/um^2').magnitude)
        r_saddle *= ureg('um')
        I_saddle *= ureg('mW/um^2')
    except ValueError:
        plt.show()
        raise ValueError(f'Could not find saddle point. Increase {beam._dmax=}.')

    if verbose:
        plt.plot(r_saddle.to('um').magnitude, I_saddle.to('mW/um^2').magnitude, 'o')
        plt.show()
    
    return r_saddle


def find_intensity_saddle_point(beam, verbose: bool = False) -> tuple:
    """Function for finding position of the saddle point of the intensity profile of a bottle beam.

    Since a bottle beam is symmetric around the axis defined by the propagation direction of the beam, 
    the saddle point can be found in the plane parallel to the k-vector of the beam.
    For example, for a beam propagating along z, the saddle point can be found in the xz-plane (or yz).
    In general, this plane is defined by the beam coordinates (z, r_perp) which are rewritten in 
    polar coordinates (r, phi).
    
    # Arguments
    * beam::BottleBeam - The beam.
    * verbose::bool - Additional output.

    # Returns
    (r_saddle::float, phi_saddle::float) - Distance and direction of the saddle point relative to the focus.
    """

    phi_saddle = _find_saddle_point_direction(beam, verbose)
    r_saddle = _find_saddle_point_radius(beam, phi_saddle, verbose)

    logging.debug(f'find_intensity_saddle_point: Found saddle point ({r_saddle=}, {phi_saddle=}).')

    return (r_saddle, phi_saddle)