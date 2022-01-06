import numpy as np
from scipy.special import jv
import scipy.integrate as integrate

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