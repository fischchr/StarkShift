from re import L
import numpy as np
from pint.registry import UnitRegistry

from sympy import sqrt
from sympy.physics.wigner import clebsch_gordan, wigner_6j

from matplotlib import pyplot as plt

from atomphys import State
from .AxialBeams import AxialBeam
from arc import AlkaliAtom

from .AlkaliCorePolarizability import alpha_j, polarization_factor, sympify_angular_momentum
from .AlkaliRydbergPolarizability import evaluate_wave_function, get_radial_integral
from .SphericalExpansion import SphericalBeamExpansion


### Helper functions ###
def get_j1(state_c: State) -> float:
    """Get the angular momentum of the core electron. 
    
    # Arguments
    * state_c::atomphys.State - State of the core electron.
    """

    return sympify_angular_momentum(state_c.J)


def get_j2(state_r: tuple) -> float:
    """Get the angular momentum of the Rydberg electron. 

    # Arguments
    * state_r::tuple(5) - Rydberg state (n, s, l, j, mj).
    """

    _, _, _, j2, _ = state_r
    return sympify_angular_momentum(j2)


### Core electron calculations ###
def alkaline_earth_core_ac_stark(state_c: State, state_r: tuple, j: float, mj: float, epsilon: str, beam: AxialBeam) -> float:
    """Calculate the contribution on the ac Stark shift from the core electron in SI units. 
    
    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * state_r::tuple(5) - Rydberg state (n, s, l, j, mj).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    * beam::AxialBeam - Trapping beam object.
    """

    # Get the unit registry
    ureg = beam.units

    # Evaluate the intensity at the position of the atom i.e. (0, 0, 0)
    r = np.array([0, 0, 0]) * ureg('m')
    I = beam.I(*r)[0]

    # Note that the polarizability is defined here with an additional minus sign compared to eq. (3.40)
    return - I / (2 * ureg('c * epsilon_0')) * alkaline_earth_core_polarizability(state_c, state_r, j, mj, beam.omega, epsilon, ureg)


def alkaline_earth_core_polarizability(state_c: State, state_r: tuple, j: float, mj: float, omega: float, epsilon: str, ureg: UnitRegistry) -> float:
    """Calculate the polarizability of the core electroin SI units (e * a0 / (V/cm)) based on eq. (3.40).
    The polarizability (eq. 3.39) is evaluated using the alkali atom function `alpha_j`. 
    
    Note that `alpha_j` has the opposite sign compared to literature.
    Therefore, an additional minus sign is introduced here s.t. the ac Stark shift of the core electron is 
        Delta E = - |E|^2 / 4 * alkaline_earth_core_polarizability
    i.e., there is an additional pair of minus signs compared to eq. (3.39) and (3.50). 

    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * state_r::tuple(5) - Rydberg state (n, s, l, j, mj).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * omega::float - Angular frequency of the laser in SI units.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    * ureg::Unitregistry - Unit registry.
    """
    
    # Get the angular momentum of the core electron
    j1 = get_j1(state_c)

    # Get the angular momentum of the Rydberg electron
    j2 = get_j2(state_r)

    # Symifpy the total angular momentum
    j = sympify_angular_momentum(j)
    mj = sympify_angular_momentum(mj)
    
    # Allocate memory for the sum
    res = 0
    
    # Sum over all k values
    for k in range(3):
        sign = 1 if (j1 + j2 + j + k) % 2 == 0 else -1
        res += sign * sqrt(2*j + 1) * \
                clebsch_gordan(j, k, j, mj, 0, mj) * \
                wigner_6j(j1, k, j1, j, j2, j) * \
                polarization_factor(epsilon, k) * \
                alpha_j(state_c, omega, k, ureg)
    
    # Note that the polarizability is defined here with an additional minus sign compared to eq. (3.40)
    # This minus sign is added here and in `alkaline_earth_core_ac_stark`
    return -float(res) * ureg('e') * ureg('a_u_length') / ureg('a_u_electric_field')


### Rydberg electron functions ###
def alkaline_earth_rydberg_ac_stark(state_c: State, state_r: tuple, j: float, mj: float, beam_expansion: SphericalBeamExpansion,
                                    arc_atom: AlkaliAtom = None, r_v: np.ndarray = None, R_eval: np.ndarray = None) -> float:
    """Calculate the contribution on the ac Stark shift from the Rydberg electron in SI units. 
    
    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * state_r::tuple(5) - Rydberg state (n, s, l, j, mj).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam_expansion::SphericalBeamExpansion - Expansion of the beam in spherical harmonics.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
                                 If not `arc_atom` is passed, `r_v` and `R_eval` must be given.
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    """

    # Get the angular momentum of the core electron
    j1 = get_j1(state_c)

    # Get the quantum numbers of the Rydberg state
    _, s2, l2, j2, mj2 = state_r

    # Make sure 3j and 6js are evaluated correctly
    s2 = sympify_angular_momentum(s2)
    j2 = sympify_angular_momentum(j2)
    mj2 = sympify_angular_momentum(mj2)

    # Symifpy the total angular momentum
    j = sympify_angular_momentum(j)
    mj = sympify_angular_momentum(mj)

    # Get the unit registry
    ureg = beam_expansion.units

    # Integrate wave function if the wave function hasn't been evaluated yet
    if arc_atom is not None:
        r_v, R_eval = evaluate_wave_function(state_r, arc_atom, ureg)

    # Allocate memory for the sum
    res = 0

    # Use eq. (B.6) to evaluate the ac Stark shift
    k_max = beam_expansion.L_max
    # Avoid unnecessary evaluation since <j mj; k 0 | j mj> vanishes for k > 2j
    k_max = np.min([k_max, float(2*j) + 1])

    # Sum over all values of k
    for k in range(int(k_max)):
        # Calcualte the sign factor
        k_sign = 1 if (j1 + j2 + j + k + l2 + s2 + j + k) % 2 == 0 else -1

        # Calculate the prefactor
        prefactor = float(
            k_sign * np.sqrt(float((2*j + 1) * (2*l2 + 1))) * (2*j2 + 1) * \
            wigner_6j(j2, k, j2, j, j1, j) * \
            wigner_6j(j2, l2, s2, l2, j2, k) * \
            clebsch_gordan(j, k, j, mj, 0, mj) * \
            clebsch_gordan(l2, k, l2, 0, 0, 0)
        )
        # Add kth sum term
        res += prefactor * get_radial_integral(r_v, R_eval, beam_expansion, k)        

    # Calculate polarizability of a free electron
    omega = 2 * np.pi * ureg.c / beam_expansion.beam.lam
    alpha_p = -1 * ureg('e')**2 / (ureg('m_e') * omega**2)

    return - (alpha_p / (2 * ureg('c') * ureg('epsilon_0')) * res).to('eV')


def alkaline_earth_rydberg_polarizability(state_c: State, state_r: tuple, j: float, mj: float, beam_expansion: SphericalBeamExpansion,
                                    arc_atom: AlkaliAtom = None, r_v: np.ndarray = None, R_eval: np.ndarray = None) -> float:
    """Calculate the polarizability of a Rydberg electron in SI units.
    
    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * state_r::tuple(5) - Rydberg state (n, s, l, j, mj).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam_expansion::SphericalBeamExpansion - Expansion of the beam in spherical harmonics.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
                                 If not `arc_atom` is passed, `r_v` and `R_eval` must be given.
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    """

    if arc_atom is None:
        # Make sure both r_v and R_eval exist if no arc_atom object is passed in
        assert not (r_v is None or R_eval is None) 
        # Make sure they have the same length
        assert r_v.shape == R_eval.shape

    # Get the unit registry
    ureg = beam_expansion.units

    # Calculate the ac Stark shift
    U = alkaline_earth_rydberg_ac_stark(state_c, state_r, j, mj, beam_expansion, arc_atom, r_v, R_eval)
    # Calculate the electric field squared
    E0_sq = 2 * beam_expansion.beam.I0 / (ureg('c') * ureg('epsilon_0'))
    # Calculate the polarizability alpha
    return - (4 * U / E0_sq).to('e * a_u_length / a_u_electric_field')


# Total stark shift
def alkaline_earth_ac_stark(state_c: State, state_r: tuple, j: float, mj: float, 
                            beam: AxialBeam, epsilon:str, N_r: int = 100, N_theta: int = 250, L_max: int = 15, 
                            arc_atom: AlkaliAtom = None, r_v: np.ndarray = None, R_eval: np.ndarray = None):
    """Calculate the ac Stark shift in SI units (J). 

    # Arguments
    * state_c::State - atomphys.State object of the core state.
    * state_r::tuple(5) - Rydberg state (n_2, s_2, l_2, j_2, mj_2).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam::AxialBeam - Representation of the beam.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    * N_r::int - Number of radial grid points.
    * N_theta::int - Number of grid points (altitude). The number of azimuthal grid points N_phi = 2 * N_theta
    * L_max::int - Order of expansion in spherical harmonics Y_{l,m}.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
                                 If not `arc_atom` is passed, `r_v` and `R_eval` must be given.
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    """
    
    # Get the unit registry
    ureg = beam.units

    # Get the expansion limits
    r_i = 1 * ureg('a0')
    r_o = 2 * beam.w0

    beam_expansion = SphericalBeamExpansion(beam, N_r, N_theta, L_max, r_i, r_o)

    # Calculate the contribution of the core electron
    U_core = alkaline_earth_core_ac_stark(state_c, state_r, j, mj, epsilon, beam)
    # Calculate the contribution of the Rydberg electron
    U_rydberg = alkaline_earth_rydberg_ac_stark(state_c, state_r, j, mj, beam_expansion, arc_atom, r_v, R_eval)

    return U_core + U_rydberg


if __name__ == '__main__':
    pass 
    
    
