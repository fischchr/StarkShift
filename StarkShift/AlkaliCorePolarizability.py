from atomphys import Atom, Laser, Transition, State
import numpy as np
from sympy import sympify, sqrt
from sympy.physics.wigner import clebsch_gordan, wigner_6j
from .PolarizationUtil import get_sph_components, get_polarization_vector, sph_to_cart
from .AxialBeams import AxialBeam

### Helper functions ###
def sympify_angular_momentum(j: float) -> float:
    """Turn an angular momentum into a rational number. """

    return sympify(f'{2 * j:.0f} / 2')

    
def polarization_factor(epsilon_sph: np.ndarray, k: int) -> float:
    """Get the spherical vector component {\epsilon^* \otimes \epsilon}_{k, 0}.
    
    # Arguments
    * epsilon_sph::vector(3) - The polarization vector in spherical coordinates w.r.t. the quantization axis of the atom.
    * k::int - Rank of the vector.
    """
    
    # Get the components of the polarization vector
    eps_m, eps_0, eps_p = get_sph_components(epsilon_sph)
    
    if k == 0:
        return -1 / np.sqrt(3)
    elif k == 1:
        return (eps_p**2 - eps_m**2) / np.sqrt(2)
    elif k == 2:
        return (3 * eps_0**2 - 1) / np.sqrt(6)
    else:
        raise ValueError(f'{k} is not a valid input.')


def get_reduced_matrix_element_sq(transition: Transition, ureg) -> float:
    """    ***NOT USED ANYMORE***
    Calculate the reduced dipole matrix element of a transition in SI units. 
    
    # Arguments
    * transition::Transition - atomphys.Transition object.
    * ureg::UnitRegistry - Unit registry.
    """
    
    # Get the energy of the states (i is always the lower energy state)
    omega_i = transition.i.energy / ureg('hbar')
    omega_f = transition.f.energy / ureg('hbar')
    # Get the decay rate
    Gamma = transition.Gamma
    # Get the angular momentum of the upper state
    j_p = transition.f.J
    
    return 3*np.pi * ureg('hbar') * ureg('epsilon_0') * ureg('c')**3 * Gamma * (2*j_p + 1) / (omega_f - omega_i)**3
    
        
def alpha_j(state_i: State, omega: float, k: int, ureg):
    """Calculate the polarizability as defined by eq. (3.21) in atomic units. 

    Note that alpha_j has the opposite sign to the usual polarizability, i.e., with this definition the ac Stark shift is E^2 / 4 * alpha_j.
    
    # Arguments 
    * state_i::State - atomphys.State object of the state of interest.
    * omega::float - Angular frequency of the laser in SI units.
    * ureg::Unitregistry - Unit registry.
    """
    
    # Get the angular momentum of state_i
    j_i = sympify_angular_momentum(state_i.J)
    # Get the energy of state_i
    E_i = state_i.energy
    
    res = 0

    # Sum over all transitions
    for transition in state_i.transitions:
        # Get the other state (in atomphys the lower energy state is always transition.i)
        if transition.i == state_i:
            state_f = transition.f
        else:
            state_f = transition.i
            
        # Get the angular momentum of state_f
        j_f = sympify_angular_momentum(state_f.J)
        # Get the energy of state_f in SI units
        E_f = state_f.energy
                
        # Sign factor (-1)^{1 + j + j'}
        sign = 1 if (1 + j_i + j_f) % 2 == 0 else -1
        
        # Get the reduced dipole matrix element  squared in atomic units
        d_au = transition.d.to('e * a_0').magnitude
        d_sq_au = np.abs(d_au**2)
        
        # Get the energy factors in atomic units
        E_m = 1 / (E_f - E_i - ureg('hbar') * omega).to('a_u_energy').magnitude
        E_p = 1 / (E_f - E_i + ureg('hbar') * omega).to('a_u_energy').magnitude
        E_p_sign = -1 if k % 2 == 1 else 1
        E = (E_m + E_p_sign * E_p)
        
        # Sum the contributions
        res += sign * wigner_6j(j_i, k, j_i, 1, j_f, 1) * d_sq_au * E
        
    return float(sqrt(2*k + 1) * res)


def alkali_core_polarizability(state_i: State, mj: float, beam: AxialBeam, epsilon: str, e_q_xyz: np.ndarray = None) -> float:
    """Calculate the polarizability in SI units using eq. (3.21)
    The unit of the polarizability is (e * a0 / (V/cm)) = (e * a0)^2 / E_h.

    The ac Stark shift is 
        Delta E = - |E|^2 / 4 * alkali_core_polarizability
    i.e., there is an additional pair of minus signs compared to eq. (3.20). 
    This minus sign is introduced here to be consistent with literature.
    
    # Arguments
    * state_i::State - atomphys.State object of the state of interest.
    * mj::float - Magnetic quantum number of the state.
    * beam::AxialBeam - Representation of the laser beam.
    * epsilon::str - Description of the polarization vector. 
    * e_q_xyz::vector(3) - Quantization axis in carthesian coordinates. If not defined, the quantization axis is set by the k-vector of the laser.
    """

    # Calculate the polarization vector in the spherical coordinates
    if e_q_xyz is None:
        # The quantization axis is defined by the laser beam
        e_p_sph = get_polarization_vector(epsilon, beam.k_hat)

        # Calculate the angle between the polarization vector and the k-vector
        theta = np.arccos(np.dot(sph_to_cart(e_p_sph), beam.k_hat))

        # Check that the polarization vector is not parallel to the k-vector
        if theta < 1e-6:
            print('Unphysical longitudinal polarization.')
    else:
        # The quantization axis is defined by e_q_xyz
        e_p_sph = get_polarization_vector(epsilon, e_q_xyz)
    
    # Get the unit registry
    ureg = beam.units
    
    # Get the angular momentum of state_i
    j_i = sympify(f'{2*state_i.J:.0f} / 2')
    
    res = 0
    
    # Sum over all k values
    for k in range(3):
        res += clebsch_gordan(j_i, k, j_i, mj, 0, mj) / sqrt(2*j_i + 1) * \
               polarization_factor(e_p_sph, k) * \
               alpha_j(state_i, beam.omega, k, ureg)

    # alpha_j has the opposite sign to the usual definition.
    return - float(res) * ureg('e') * ureg('a_u_length') / ureg('a_u_electric_field')


def alkali_core_ac_stark(state_i: State, mj: float, beam: AxialBeam, epsilon: str, e_q: np.ndarray = None):
    """Calculate the ac Stark shift in SI units (J) using eq. (3.20)

    # Arguments
    * state_i::State - atomphys.State object of the state of interest.
    * mj::float - Magnetic quantum number of the state.
    * beam::AxialBeam - Representation of the laser beam.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    """

    # Get the unit registry
    ureg = beam.units
    
    # You could calculate the electric field 
    #E_0 = np.sqrt(2 * I / (ureg('epsilon_0') * ureg('c')))
    
    # Or use the formula for the Stark shift with the intensity
    # Note the minus sign here and in alkali_core_polarizability compared to eq. (3.20) and (3.21).
    return - beam.I0 / (2 * ureg('c * epsilon_0')) * alkali_core_polarizability(state_i, mj, beam, epsilon, e_q)


if __name__ == "__main__":
    pass