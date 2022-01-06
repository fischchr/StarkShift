import pint 
import numpy as np
from arc import Potassium

from StarkShift import GaussianBeam, SphericalBeamExpansion
from StarkShift import alkali_rydberg_polarizability, alkali_rydberg_ac_stark

if __name__ == '__main__':
    # Initialize unit registry
    ureg = pint.UnitRegistry()

    # Define the ARC atom
    arc_atom = Potassium()
    # Define the Rydberg state (n, s, l, j, mj)
    state_r = (50, 1/2, 0, 1/2, 1/2)

    # Define the laser properties
    P = 100 * ureg('mW')                    # Beam power
    lam = 1180 * ureg('nm')                 # Wavelength
    k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
    w0 = 2 * ureg('um')                     # Beam waist
    r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

    # Define the laser beam
    beam = GaussianBeam(P, k, w0, r0, ureg)

    # Expand the beam in spherical harmonics
    beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o =2 * w0)

    # Calculate the polarizability
    alpha = alkali_rydberg_polarizability(state_r, beam_expansion, arc_atom)

    # Or calculate the ac Stark shift
    U = alkali_rydberg_ac_stark(state_r, beam_expansion, arc_atom)

    # Get the ac Stark shift in mK
    U_kB = (U / ureg('k_B')).to('mK')

    print(U_kB)