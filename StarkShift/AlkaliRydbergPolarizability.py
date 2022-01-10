from arc.alkali_atom_data import Rubidium
import numpy as np
from sympy.physics.wigner import wigner_6j, clebsch_gordan
from arc import AlkaliAtom
from pint import UnitRegistry
from .AlkaliCorePolarizability import sympify_angular_momentum
import logging

from .SphericalExpansion import SphericalBeamExpansion



### Helper functions ###

def get_radial_integral(r_v: np.array, R_eval: np.array, interpolated_beam: SphericalBeamExpansion, k: int) -> float:
    """Calculate the radial integral <n l | I_{k, 0} | n l > in SI units using eq. (3.28).

    # Arguments
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    * interpolated_beam::SphericalBeamExpansion - Expansion of the beam in spherical harmonics.
    * k::int - Rank of the operator I_{k, 0}.
    """

    # Get the unit registry
    ureg = interpolated_beam.units

    # Interpolate the exapnsion coefficients on all points of the grid we evaluate the wavefunction on
    fkqs = interpolated_beam.get_coeff(r_v, k, 0) # shape = (N_R,)

    # Evaluate the product of the wavefunctions on r_v
    # The unit of R is (a0^(-3/2))
    R_sq = R_eval**2

    # Calculate the integrand in the r integral for each point in r_v. 
    # The result must have units of intensity
    integrand = r_v[:-1].to('a_u_length')**2 * \
                np.diff(r_v).to('a_u_length') * \
                R_sq[:-1].to('a_u_length^(-3)') * \
                fkqs[:-1].to('a_u_energy / (a_u_time * a_u_length^2)')

    # Remove all imaginary residuals which might be present
    integrand = np.real_if_close(integrand.to('a_u_energy/(a_u_time * a_u_length^2)').magnitude) * ureg('a_u_energy/(a_u_time * a_u_length^2)')

    # Return the integral
    return np.sum(integrand)


def evaluate_wave_function(configuration_i: tuple, arc_atom: AlkaliAtom, ureg: UnitRegistry) -> tuple:
    """Evaluate the radial wave function in SI units. 
    
    # Arguments
    * configuration_i::tuple(4) - Rydberg configuration (n, s, l, j).
    * arc_atom::arc.AlkaliAtom - Representation of the atom from ARC
    * ureg::Unitregistry - Unit registry.

    # Returns
    * r_v::np.array(N) - Radial grid points (in SI units).
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v (in SI units).
    """

    logging.debug(f'evaluate_wave_function: {configuration_i=}, {arc_atom=}')

    # Get the quantum numbers of the state
    n, s, l, j = configuration_i

    # Define the grid on which to evaluate
    a_i = 10 * ureg('a0')
    a_o = 2*n * (n + 15) * ureg('a0')
    step = 0.001

    # Evaluate the radial wavefunction on the grid
    r_v, rho = arc_atom.radialWavefunction(
        l, s, j,\
        arc_atom.getEnergy(n, l, j, s) * ureg('eV').to('a_u_energy').magnitude,\
        a_i.to('a_u_length').magnitude,\
        a_o.to('a_u_length').magnitude, 
        step)

    # Remove the zero at the beginning
    r_v = r_v[1:] * ureg('a_u_length')
    R_eval = rho[1:] * ureg('a_u_length * a_u_length^(-3/2)') / r_v

    return r_v, R_eval


def alkali_rydberg_ac_stark(state_i: tuple, beam_expansion: SphericalBeamExpansion,
                     arc_atom: AlkaliAtom = None, r_v: np.ndarray = None, R_eval: np.ndarray = None) -> float:

    """Calculate the ac Stark shift of a Rydberg electron in SI units using eq. (3.27).

    # Arguments
    * state_i::tuple(5) - Rydberg state (n, s, l, j, mj).
    * beam_expansion::SphericalBeamExpansion - Expansion of the beam in spherical harmonics.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
                                 If not `arc_atom` is passed, `r_v` and `R_eval` must be given.
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    """

    logging.debug(f'alkali_rydberg_ac_stark: {state_i=}, {arc_atom=}')

    # Get the quantum numbers of the state
    n, s, l, j, mj = state_i

    # Make sure 3j and 6js are evaluated correctly
    s = sympify_angular_momentum(s)
    j = sympify_angular_momentum(j)
    mj = sympify_angular_momentum(mj)

    # Log the values
    logging.debug(f'Got angular momenta: {s=}, {l=}, {j=}, {mj=}')

    # Get the unit registry
    ureg = beam_expansion.units

    # Evaluate the wave function if the wave function hasn't been evaluated yet
    if arc_atom is not None:
        # Get the configuration of the state
        configuration_i = (n, s, l, j)
        # Evaluate the wave function
        r_v, R_eval = evaluate_wave_function(configuration_i, arc_atom, ureg)

    # Allocate memory for the sum
    res = 0

    # Use eq. (3.27) to evaluate the ac Stark shift
    k_max = beam_expansion.L_max
    # Avoid unnecessary evaluation since <j mj; k 0 | j mj> vanishes for k > 2j
    k_max = np.min([k_max, float(2*j) + 1])
    logging.debug(f'Summing k up to {k_max=}')

    for k in range(int(k_max)):
        # Calcualte the sign factor
        k_sign = 1 if (l + s + j + k) % 2 == 0 else -1

        # Calculate the prefactor
        prefactor = float(
            k_sign * np.sqrt(float((2*j + 1) * (2*l + 1))) * \
            wigner_6j(j, l, s, l, j, k) * \
            clebsch_gordan(j, k, j, mj, 0, mj) * \
            clebsch_gordan(l, k, l, 0, 0, 0)
        )
        # Calcualte the radial integral
        R_int = get_radial_integral(r_v, R_eval, beam_expansion, k)

        # Add kth sum term
        res += prefactor * R_int

        # Log the result
        logging.debug(f'Sum terms: {k=}, {k_sign=}, {prefactor=}, {R_int=}') 

    # Calculate polarizability of a free electron
    omega = 2 * np.pi * ureg('c') / beam_expansion.beam.lam
    alpha_p = -1 * ureg('e')**2 / (ureg('m_e') * omega**2)

    return - (alpha_p / (2 * ureg('c') * ureg('epsilon_0')) * res).to('eV')


def alkali_rydberg_polarizability(state_i: tuple, beam_expansion: SphericalBeamExpansion,
                                 arc_atom: AlkaliAtom = None, r_v: np.ndarray = None, R_eval: np.ndarray = None) -> float:
    """Calculate the polarizability of a Rydberg electron in SI units.

    # Arguments
    * state_i::tuple(5) - Rydberg state (n, s, l, j, mj).
    * beam_expansion::SphericalBeamExpansion - Expansion of the beam in spherical harmonics.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
                                 If not `arc_atom` is passed, `r_v` and `R_eval` must be given.
    * r_v::np.array(N) - Radial grid points.
    * R_eval::np.array(N) - Radial wave function R_{nl}(r_v) evaluated on r_v.
    """

    logging.debug(f'alkali_rydberg_polarizability: {state_i=}, {arc_atom=}')

    if arc_atom is None:
        # Make sure both r_v and R_eval exist if no arc_atom object is passed in
        assert not (r_v is None or R_eval is None) 
        # Make sure they have the same length
        assert r_v.shape == R_eval.shape

    # Get the unit registry
    ureg = beam_expansion.units

    # Calculate the ac Stark shift
    U = alkali_rydberg_ac_stark(state_i, beam_expansion, arc_atom, r_v, R_eval)
    # Calculate the electric field squared
    E0_sq = 2 * beam_expansion.beam.I0 / (ureg('c') * ureg('epsilon_0'))
    # Calculate the polarizability alpha
    return - (4 * U / E0_sq).to('e * a_u_length / a_u_electric_field')


if __name__ == '__main__':
    pass

    
