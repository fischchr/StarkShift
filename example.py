import pint
import numpy as np
from arc import Hydrogen
from atomphys import Atom

from StarkShift import GaussianBeam, alkaline_earth_ac_stark_jj

# Make a unit registry
ureg = pint.UnitRegistry()

# Choose the core state
ap_atom = Atom('Ca+', ureg=ureg)
state_c = ap_atom('S1/2') # 4s S_1/2

# Choose a Rydberg state
arc_atom = Hydrogen()
state_r = (50, 0, 49, 49, 49) # (n, s, l, j, mj)

# Choose a total angular momentum 
j = 49 + 1/2
mj = j

# Define a trapping beam
P = 100 * ureg('mW')                    # Power
lam = 1180 * ureg('nm')                 # Wavelength
k = 2*np.pi / lam * np.array([0, 0, 1]) # k-vector
w0 = 2 * ureg('um')                     # Waist
r0 = np.array([0, 0, 0]) * ureg('mm')   # Position of the waist

beam = GaussianBeam(P, k, w0, r0, ureg)

# Define a polarization in spherical coordinates
epsilon = [0, 1, 0]                     # (e_p, e_0, e_p)

# Check that core shift is the same as for an alkali atom
state_r_0 = (50, 0, 0, 0, 0)
j_0 = 1/2
mj_0 = j_0
U_AE = alkaline_earth_core_ac_stark_jj(state_c, state_r_0, 1/2, 1/2, epsilon, beam)
from AlkaliCorePolarizability import alkali_core_ac_stark
U_A = alkali_core_ac_stark(state_c, mj_0, beam.I0, beam.omega, epsilon, ureg)

#print(U_AE.to('eV'), U_A.to('eV'))
assert np.abs(U_AE - U_A) < 1e-10 * ureg('eV')

# Check that the Rydberg state shift is the same as for an alkali atom
ap_atom_0 = Atom('Ca')
# We need a state with j1 = 0
state_c0 = ap_atom_0('1S0')
# Total angular momentum is just the one of the Rydberg electron
j_0 = state_r[3]
mj_0 = j_0

beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o = 5 * ureg('um'))
U_AE = alkaline_earth_rydberg_ac_stark(state_c0, state_r, j_0, mj_0, beam_expansion, arc_atom)
from AlkaliRydbergPolarizability import akali_rydberg_ac_stark
U_A = akali_rydberg_ac_stark(state_r, beam_expansion, arc_atom)

#print(U_AE.to('eV'), U_A.to('eV'))
assert np.abs(U_AE - U_A) < 1e-10 * ureg('eV')

# Expand the beam in spherical harmonics
beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, r_i=1 * ureg('a0'), r_o = 5 * ureg('um'))

# Get the polarizability
print(alkaline_earth_ac_stark_jj(state_c, state_r, j, mj, beam, epsilon, N_r=100, N_theta=250, L_max=15, arc_atom=arc_atom).to('eV'))