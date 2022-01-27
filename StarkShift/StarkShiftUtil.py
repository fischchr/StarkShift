import logging
import numpy as np


from .SphericalExpansion import SphericalBeamExpansion
from .AlkalineEarthPolarizability import alkaline_earth_ac_stark
from .BottleBeamUtils import find_intensity_saddle_point


def _evaluate_potential(rs, phi, state_c, arc_atom, state_r, j, mj, beam, epsilon) -> np.ndarray:
    """Function for evaluating the bottle beam potential at a certain angle phi.

    The plane (z, r_perp), which is defined by the k-vector of the beam, can be transformed to 
    polar coordinates (r, phi). The potential of the bottle beam U(r, phi) is then evaluated for
    the values rs.

    # Arguments
    * rs::vector(N) - Radial coordinates
    * phi::float - Direction along which to evaluate the potential.
    * state_c::atomphys.State - State of the core electron.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
    * state_r::tuple(4) - Rydberg state (n, s, l, j).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam::BottleBeam - The bottle beam.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.

    # Returns
    U::vector(N) - The potential of the bottle beam U(rs, phi) evaluated for all values rs.
    """

    # Get the unit registry
    ureg = beam.units
    
    # Define the coordinates relative to the focal point of the beam
    zs = rs * np.cos(phi)
    xs = rs * np.sin(phi)
    ys = np.zeros(rs.shape)
    
    # Positions at which to evaluate the potential
    r0s = np.stack((xs, ys, zs), axis=1)
    assert r0s.shape == (len(rs), 3)
    assert r0s[0].shape == (3,)
    
    # Allocate memory for the result
    U = np.zeros(rs.shape) * ureg('eV')
    
    for i, r0 in enumerate(r0s):
        # Define the beam
        beam.r0 = r0
        
         # Expand the beam in spherical harmonics
        beam_expansion = SphericalBeamExpansion(beam, N_r=100, N_theta=250, L_max=15, \
                                                r_i=1*ureg('a0'), r_o =20*ureg('um'))
        # Evaluate the Stark shift
        U[i] = alkaline_earth_ac_stark(state_c, state_r, j, mj, beam_expansion, epsilon, arc_atom=arc_atom)
        
    return U


def find_trap_depth(state_c, arc_atom, state_r, j, mj, beam, epsilon, verbose: bool = False) -> tuple:
    """Find the trap depth for an alkaline-earth Rydberg atom in a bottle beam trap. 

    This function first finds the direction and distance of the saddle point of the optical intensity relative
    to the focus of the beam. This saddle point in optical intensity is identical to the saddle point of the 
    optical potential. This can be verified using the function `_find_trap_depth_depreciated`.

    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
    * state_r::tuple(4) - Rydberg state (n, s, l, j).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam::BottleBeam - The bottle beam.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    * verbose::bool - Additional output.

    # Returns
    (r_saddle, phi_saddle, U_saddle, trap_depth) - Position of the saddle point, height of the saddle point and trap depth.
    """

    # Get the unit registry
    ureg = beam.units

    # Find the distance and direction of the saddle point
    r_saddle, phi_saddle = find_intensity_saddle_point(beam, verbose)

    # Calculate the trap depth
    U_center = _evaluate_potential(np.zeros(1) * ureg('um'), 0, state_c, arc_atom, state_r, 
                                   j, mj, beam, epsilon)[0]
                                   
    U_saddle = _evaluate_potential(np.linspace(r_saddle, 2*r_saddle, 1), phi_saddle, state_c, arc_atom, state_r, 
                                   j, mj, beam, epsilon)[0]

    
    trap_depth = ((U_saddle - U_center) / ureg('k_B')).to('mK')

    logging.debug(f'find_trap_depth: Found trapping saddle point ({r_saddle=}, {phi_saddle=}). Potential {U_saddle=}, {trap_depth=}.')
    
    return r_saddle, phi_saddle, U_saddle, trap_depth


def _find_trap_depth_depreciated(state_c, arc_atom, state_r, j, mj, beam, epsilon, verbose: bool = False) -> tuple:
    """Find the trap depth for an alkaline-earth Rydberg atom in a bottle beam trap. 

    NOT USED ANYMORE.

    This function first finds the direction of the saddle point of the optical intensity. The direction of the
    minimum of the optical trapping potential must be in the same direction as the direction of the saddle point 
    of the optical intensity. However, the position might be different due to the extent of the Rydberg wave function.
    Therefore, in the second step the ac Stark shift is evaluated radially along the direction of the saddle point
    of the optical intensity.

    # Arguments
    * state_c::atomphys.State - State of the core electron.
    * arc_atom::arc.AlkaliAtom - Representation of the atom used for numerically evaluating the radial wave function.
    * state_r::tuple(4) - Rydberg state (n, s, l, j).
    * j::float - Total angular momentum j.
    * mj::float - Total magnetic quantum number m_j.
    * beam::BottleBeam - The bottle beam.
    * epsilon::str - Description of the polarization vector. See PolarizationUtil.evaluate_vector_description for details.
    * verbose::bool - Additional output.

    # Returns
    (r_saddle, phi_saddle, U_saddle, trap_depth) - Position of the saddle point, height of the saddle point and trap depth.
    """

    from .BottleBeamUtils import _find_maximum

    # Get the unit registry
    ureg = beam.units

    # Find the distance and direction of the saddle point
    r_saddle, phi_saddle = find_intensity_saddle_point(beam, verbose)

    # Calculate the potential around the saddle point
    dr = 0.5 * ureg('um')
    rs = np.linspace(r_saddle - dr, r_saddle + dr, 10)
    U = _evaluate_potential(rs, phi_saddle, state_c, arc_atom, state_r, j, mj, beam, epsilon)

    if verbose:
        # Plot the second result
        from matplotlib import pyplot as plt
        plt.plot(rs.to('um').magnitude, U.to('eV').magnitude)
        plt.xlabel('$r$ (\u00B5m)')
        plt.ylabel(f'$U(r)$ (eV)')
        plt.show()

    try:
        r_saddle, U_saddle = _find_maximum(rs.to('um').magnitude, U.to('eV').magnitude)
        r_saddle *= ureg('um')
        U_saddle *= ureg('eV')
    except ValueError:
        raise ValueError(f'Could not find saddle point. Increase {beam._dmax=}.')
    
    # Calculate the trap depth
    U_center = _evaluate_potential(np.zeros(3) * ureg('um'), phi_saddle, state_c, arc_atom, state_r, 
                                   j, mj, beam, epsilon)[0]
    
    trap_depth = ((U_saddle - U_center) / ureg('k_B')).to('mK')

    logging.debug(f'find_trap_depth: Found trapping saddle point ({r_saddle=}, {phi_saddle=}). Potential {U_saddle=}, {trap_depth=}.')
    
    return r_saddle, phi_saddle, U_saddle, trap_depth