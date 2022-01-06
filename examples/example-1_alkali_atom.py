import pint 
import numpy as np
from atomphys import Atom

from StarkShift import alkali_core_ac_stark
from StarkShift import GaussianBeam
from StarkShift import alkali_core_polarizability, alkali_core_ac_stark

if __name__ == '__main__':
    # Initialize unit registry
    ureg = pint.UnitRegistry()

    # Define the atomphys atom
    ap_atom = Atom('Ca', ureg=ureg)
    # Define the atomphys state
    ap_state = ap_atom('1S0')
    # Define the mj value
    mj1 = 0

    # Define the laser properties
    P = 100 * ureg('mW')                    # Beam power
    lam = 1180 * ureg('nm')                 # Wavelength
    k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
    w0 = 2 * ureg('um')                     # Beam waist
    r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

    # Define the laser beam
    beam = GaussianBeam(P, k, w0, r0, ureg)

    # Define the polarization
    epsilon = 'x'

    # Calculate the polarizability
    alpha = alkali_core_polarizability(ap_state, mj1, beam, epsilon)

    # Or calculate the ac Stark shift
    U = alkali_core_ac_stark(ap_state, mj1, beam, epsilon)

    # Get the ac Stark shift in mK
    U_kB = (U / ureg('k_B')).to('mK')

    print(U_kB)