import numpy as np

def get_w0(zR: float, lam: float) -> float:
    """Calculate the waist of a beam. Arguments must be unitful.
    
    # Arguments
    * zR::float - Rayleigh range.
    * lam::float - Wavelength.
    
    # Returns
    * w0::float - Beam waist.
    """
    
    w0 = np.sqrt(zR * lam / np.pi)
    
    return w0.to(zR.units)


def get_zR(w0: float, lam: float) -> float:
    """Calculate the Rayleigh length. Arguments must be unitful.
    
    # Arguments
    * w0::float - Waist.
    * lam::float - Wavelength.
    
    # Returns
    * zR::float - Rayleigh range.
    """
    
    zR = np.pi * w0**2 / lam
    
    return zR.to(w0.units)


def get_w(w0: float, lam: float, z: np.array) -> float:
    """Calculate the beam radius at a point z. Arguments must be unitful.
    
    # Arguments
    * w0::float - Beam waist.
    * lam::float - Wavelength.
    * z::np.array - Position at which the beam radius is calculated.

    # Returns
    * w::np.array - Beam waist w(z).
    """
    
    # Calculate Rayleigh length
    zR = get_zR(w0, lam)
    # Calculate beam radius
    w = w0 * np.sqrt(1 + (z / zR)**2)
    
    return w.to(w0.units)


def get_I0(P0: float, w0: float) -> float:
    """Calculate the beam intensity from power. Arguments must be unitful.
    
    # Arguments
    * P0::float - Beam power.
    * w0::float - Beam waist.

    # Returns
    * I0::np.array - Beam intensity at the waist.
    """
    
    I0 = 2 * P0 / (np.pi * w0**2)
    
    return I0.to('W/cm^2')


def get_P0(I0: float, w0: float) -> float:
    """Calculate the beam power from intensity. Arguments must be unitful.
    
    # Arguments
    * I0::float - Beam intensity at the waist.
    * w0::float - Beam waist.

    # Returns
    * P0::np.array - Beam power.
    """
    
    P0 = I0 * (np.pi * w0**2) / 2
    
    return P0.to('W')



if __name__ == "__main__":
    # Unit tests

    # Load unit registry
    import pint
    ureg = pint.UnitRegistry()

    # Define beam parameters
    w0 = 1 * ureg('mm')
    lam = 500 * ureg('nm')
    zR = (np.pi * w0**2 / lam).to('mm')

    P0 = 10 * ureg('mW')
    I0 = 2 * P0 / (np.pi * w0**2)

    assert zR == get_zR(w0, lam)
    assert w0 == get_w0(zR, lam)
    assert np.sqrt(2) * w0 == get_w(w0, lam, zR)
    assert I0 == get_I0(P0, w0)
    assert P0 == get_P0(I0, w0)