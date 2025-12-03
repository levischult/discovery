import jax.numpy as jnp
import numpy as np
import healpy as hp
import scipy.special as ss
import discovery.deterministic as dsd
#from discovery.deterministic import fpc_fast
from time import time_ns
import jax
from jax.scipy import linalg as jsl
import jax_healpy as jhp


"""
Levi Schult, Kyle Gersbach 2025

Discovery edition of anis_coefficients.py from enterprise.
Many functions are taken directly from enterprise:anis_coefficients.py
We have tried to explicitly state which these are using: 
'DIRECT IMPORT FROM ENTERPRISE ANIS_COEF' in the docstring. If function name is
changed, see the direct import line for the original name.

Conventions:
Khat - unit vector pointing in the direction of GW propagation (source to observer)
etahat - unit vector pointing in the direction of the GW source (observer to source)
"""

# Constants
DAY_SEC = 86400.0  # Number of seconds in a day
YEAR_SEC = (365.24 * 24 * 3600)  # Number of seconds in a year
F_YEAR_HZ = 1.0 / YEAR_SEC  # Frequency of 1/year in Hz

C_MPS = 299792458.0  # Speed of light in m/s

PC_M = 3.085677581491367e+16  # Parsec in meters
KPC_M = PC_M * 1e3  # Kiloparsec in meters
MPC_M = PC_M * 1e6  # Megaparsec in meters

T_SUN_SEC = 4.9254909476412675e-06 # Solar mass time in seconds

def fpc_fast(pos, gwtheta, gwphi):
    """Get the Fplus and Fcross antenna response functions

    This function returns a tuple containing the Fplus and Fcross antenna response
    functions for each pulsar postion and each gw source direction. This function
    supports vectorization for both positions (N, 3), and gwtheta and gwphi (M)

    Parameters
    ----------
    pos : array
        The array of pulsar position(s). shape=(3) or (N,3)
    gwtheta : scalar or array
        The gw origin polar angle(s) for etahat. shape=(1) or (M)
    gwphi : scalar or array
        The gw origin azimuthal angle(s) for etahat. shape=(1) or (M)

    Returns
    -------
    tuple
        A tuple containing (Fplus, Fcross) of shape (N,M)
    """
    # This bit makes sure the inputs are in the right shape for broadcasting
    if pos.ndim==2:
        x, y, z = pos.T # each are shape (N)
    else:
        x, y, z = pos[:,None] # each are shape (N=1)

    x, y, z = x[:,None], y[:,None], z[:,None] # Into shape (N,1) 
    gwtheta = jnp.atleast_1d(gwtheta)[None,:] # (1,M)
    gwphi = jnp.atleast_1d(gwphi)[None,:] # (1,M)

    # Helpers
    sin_phi = jnp.sin(gwphi) # (1,M)
    cos_phi = jnp.cos(gwphi) # (1,M)
    sin_theta = jnp.sin(gwtheta) # (1,M)
    cos_theta = jnp.cos(gwtheta) # (1,M)

    # Dot products
    m_dot_pos = sin_phi*x - cos_phi*y # (N,M)
    n_dot_pos = -cos_theta*cos_phi*x - cos_theta*sin_phi*y + sin_theta*z # (N,M)
    omhat_dot_pos = -sin_theta*cos_phi*x - sin_theta*sin_phi*y - cos_theta*z # (N,M)

    denom = 1.0 + omhat_dot_pos # (N,M)

    # Actual fplus and fcross calculations
    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom # (N,M)
    fcross = (m_dot_pos * n_dot_pos) / denom # (N,M)

    # Squeeze out any extra dimensions if not vectorized
    fplus = jnp.squeeze(fplus)
    fcross = jnp.squeeze(fcross)

    return fplus, fcross # Both are (N,M)

# Utility functions for other methods
def spherical2cartesian(theta, phi):
    """Convert spherical coordinates to unit-radius Cartesian coordinates.

    This function takes spherical coordinates (theta, phi) and converts them
    to Cartesian coordinates (x, y, z) on the unit sphere.

    This function supports array inputs for theta and phi.

    Parameters
    ----------
    theta : float or array_like
        The polar angle in radians.
    phi : float or array_like
        The azimuthal angle in radians.

    Returns
    -------
    jnp.ndarray
        The coordinates in an array of shape (..., 3)
    """
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.array([x, y, z]).T


def cartesian2spherical(pos):
    """Convert Cartesian coordinates to spherical coordinates.

    This function takes Cartesian coordinates (x, y, z) and converts them
    to spherical coordinates (theta, phi). This function supports vectorized
    inputs for pos.

    Parameters
    ----------
    pos : array_like
        The Cartesian coordinates. Shape (N, 3)

    Returns
    -------
    tuple
        The spherical coordinates (theta, phi). Both are of shape (N,)
    """

    x, y, z = pos.T
    r = jnp.sqrt(x**2 + y**2 + z**2)

    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x) % (2 * jnp.pi)

    return theta, phi


def pairs2matrix(M):
    # M has (npairs, N) -> return Mp with (N, npsr, npsr)
    npair = M.shape[0] # number of pairs
    nbasis = M.shape[1] # number of basis functions
    npsr = int(jnp.sqrt(0.25 + 2*npair) - 0.5) # Smaller root of quadratic formula

    Mprime = jnp.zeros((nbasis, npsr, npsr))
    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = Mprime.at[:, a, b].set(M.T)
    Mprime = Mprime.at[:, b, a].set(M.T)

    return Mprime


def matrix2pairs(M):
    # M has shape (N, npsr, npsr) -> return Mp with (npairs, N)
    npsr = M.shape[1]
    nbasis = M.shape[0]
    npair = npsr*(npsr+1)//2

    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = Mprime.at[:, :].set(M[:, a, b].T)

    return Mprime

def get_pixel_power_basis(pos, nside=16):

    npsr = pos.shape[0]
    npix = hp.nside2npix(nside)

    # Define etahat
    eta_theta, eta_phi = hp.pix2ang(nside, jnp.arange(npix))
    
    # Get Fplus and Fcross
    fp, fc = fpc_fast(pos, eta_theta, eta_phi) # Each are (N, npix)

    # Get the pulsars a and b
    a,b = jnp.triu_indices(npsr, k=0) # Includes diagonal

    # Build the power basis
    R = fp[a,:]*fp[b,:] + fc[a,:]*fc[b,:] # Shape (Npairs, npix)
    # Add normalization
    R = R * 3/(2*npix) # (Npairs, npix)

    # Convert to square covariance matrix
    Rcov = pairs2matrix(R)  # Shape (npix, npsr, npsr)

    # Double the diagonal elements for pulsar term
    idx = jnp.arange(npsr)
    Rcov = Rcov.at[:, idx, idx].set(2*Rcov[:, idx, idx])

    return Rcov  # Shape (npix, npsr, npsr)

def get_radiometer_orf(pos, nside=16):
    """
    outer function to initialize radiometer analysis of pixel power basis.

    once initialized, use inner function to get orf for any set of gwcostheta
    and gwphi values. 

    normalization is included (3/(2*npix)) and the pulsar term is accounted for
    incoherently via a doubling of pulsar auto correlations in the output ORF 
    of the inner function.
    
    As standard for discovery anisotropy functions, gwcostheta and gwphi are in 
    the convention of etahat, i.e. source direction. This is opposite to the
    convention in enterprise.signals.anis_coefficients, which uses gw propagation
    direction (khat = -etahat).

    Parameters
    ----------
    pos : array
        array of pulsar positions, shape (npsr, 3)
    nside : int, optional
        healpy nside parameter (resolution of the healpix map), by default 16
    """
    npix = hp.nside2npix(nside)

    # LSS get the pixel power response matrix
    R = get_pixel_power_basis(pos, nside) # (npix, npsr, npsr)

    # LSS multiply by Npix to put all power in each pixel individually
    # (only using one ORF at a time)
    R = R * npix

    @jax.jit
    def radiometer_orf(pos1, pos2, gwcostheta, gwphi):
        """
        return radiometer orf for a given set of gwcostheta and gwphi.

        pos1 and pos2 are not used in this function, but are included to keep 
        the function signature consistent with other orf functions.

        we use gwcostheta over gwtheta to be consistent with samplers that use
        uniform sampling in costheta.

        Remember! gwcostheta and gwphi are in the convention of etahat, i.e.
        source direction. This is opposite to the convention in
        enterprise.signals.anis_coefficients, which uses propagation direction
        (khat = -etahat).

        Parameters
        ----------
        pos1 : array
            pulsar 1 position, output from pulsar.pos -- NOT USED
        pos2 : array
            pulsar 2 position, output from pulsar.pos -- NOT USED
        gwcostheta : array
            cosine of the gravitational wave source polar angle
        gwphi : array
            phi of the gravitational wave source polar angle

        Returns
        -------
        orf : array
            radiometer orf for the given sky location
        """
        gwtheta = jnp.arccos(gwcostheta) # correcting for sampling in costheta
        # LSS get etahat pixel index
        etahat_pidx = jhp.ang2pix(nside=nside, theta=gwtheta, phi=gwphi)
        return R[etahat_pidx, :, :]
    return radiometer_orf  

def get_pixel_strain_basis(pos, nside=16):
    npsr = pos.shape[0]
    npix = hp.nside2npix(nside)

    # Define etahat
    eta_theta, eta_phi = hp.pix2ang(nside, jnp.arange(npix))

    # Get Fplus and Fcross
    fp, fc = fpc_fast(pos, eta_theta, eta_phi) # Each are (N, npix)

    F = jnp.zeros((npsr, 2*npix)) # Interleave fplus and fcross
    F = F.at[:, 0::2].set(fp)
    F = F.at[:, 1::2].set(fc)

    # Normalize this basis
    F = F * jnp.sqrt(3/(2*npix)) # (npsr, 2*npix)

    return F  # Shape (npsr, 2*npix)

def clm2alm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex

    PORT FROM ENTERPRISE.ANIS_COEFFICIENTS
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.

    Parameters
    -----------
    clm : array
        array of clm values

    Returns
    -------
    alm : array
        array of complex alm values
    """
    maxl = int(np.sqrt(len(clm))) - 1

    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl + 1):
        for mm in range(-ll, ll + 1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))

            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)

            clmindex += 1

    return alm

def clm2map(clm, nside):
    """
    generate a healpix map from clm coefficients.

    PORT FROM ENTERPRISE.ANIS_COEFFICIENTS

    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    Parameters
    ----------
    clm : array
        Array of C_{lm} values (inc. 0,0 element)
    nside : int
        Nside of the healpix pixelation (resolution parameter)

    Returns
    -------
    h : array
        Healpix pixels to be used with healpy functions
    """
    maxl = int(np.sqrt(len(clm))) - 1
    alm = clm2alm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h

def alm2clm(alm):
    """
    Given an array of complex alm values, return an array of clm values

    PORT FROM ENTERPRISE.ANIS_COEFFICIENTS

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
    Parameters
    -----------
    alm : array
        array of complex alm values

    Returns
    -------
    clm : array
        array of clm values
    """
    nalm = len(alm)
    maxl = int(np.sqrt(9.0 - 4.0 * (2.0 - 2.0 * nalm)) * 0.5 - 1.5)  # Really?
    nclm = (maxl + 1) ** 2

    # Check the solution. Went wrong one time..
    if nalm != int(0.5 * (maxl + 1) * (maxl + 2)):
        raise ValueError("Check numerical precision. This should not happen")

    clm = np.zeros(nclm)

    clmindex = 0
    for ll in range(0, maxl + 1):
        for mm in range(-ll, ll + 1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))

            if mm == 0:
                clm[clmindex] = alm[almindex].real
            elif mm < 0:
                clm[clmindex] = -alm[almindex].imag * np.sqrt(2)
            elif mm > 0:
                clm[clmindex] = alm[almindex].real * np.sqrt(2)

            clmindex += 1

    return clm

def get_linspharm_basis(pos, lmax, nside=16):
    """generate the linear spherical harmonic basis based on etahat direction
    
    etahat: origin to GW source
    khat: GW propagation direction
    enterprise_anis_coefficients.anis_basis uses khat to form the basis.
    to get an orf for a given set of clm values, it is simply: clm @ basis

    Parameters
    ----------
    pos : array
        array of pulsar positions shape (Npsr, 3)
    lmax : _type_
        lmax for spherical harmonic basis. The number of modes will be (lmax+1)^2
    nside : int, optional
        healpy resolution of sky (used for making pixel power basis), by default 16

    Returns
    -------
    basis : array
        array of shape (Nmodes, Npsr, Npsr) where Nmodes = (lmax+1)^2
        
    """
    npix = hp.nside2npix(nside)
    npsrs = pos.shape[0]

    # LSS get pixel power basis for pta
    R = get_pixel_power_basis(pos, nside)

    # LSS getting theta phi for entire sky (etahat direction)
    skypix = hp.pix2ang(nside, jnp.arange(npix), nest=False)
    gwtheta = skypix[0]
    gwphi = skypix[1]

    # LSS loop over (l, m)
    basis = []
    nclm = (lmax + 1)**2 
    clm_idx = 0
    for ll in range(0, lmax+1): # LSS lmax+1 to get lmax included
        for mm in range(-ll, ll+1):
            clm = np.zeros(nclm)
            clm[clm_idx] = 1.0 
            skymap = clm2map(clm, nside) # LSS make skymap for idx mode
            orf = R.T @ skymap # LSS make ORF from that spharm mode
            # LSS pulsar term is included in R already - see get_pixel_power_basis
            basis.append(orf)
            clm_idx += 1 # LSS move to next spharm mode

    return jnp.array(basis) # Nmodes, Npsr, Npsr

def get_linspharm_orf(pos, lmax, nside=16):
    """outer function to get linear spherical harmonic orf closure

    call to initialize linear spherical harmonic basis once, then use inner 
    function to get orf for given clm values.

    Parameters
    ----------
    pos : array
        array of pulsar positions shape (Npsr, 3)
    lmax : int
        lmax for spherical harmonic basis. The number of modes will be (lmax+1)^2
    nside : int, optional
        healpy nside parameter, by default 16
    """
    # linsph_basis is shape: Nmodes, Npsr, Npsr
    linsph_basis = get_linspharm_basis(pos, lmax, nside) 
    # LSS make set that maps psr positions to indices in correlation 
    # matrix

    @jax.jit
    def linsph_orf(pos1, pos2, clm):
        """
        return orf for given clm values in linear spherical harmonic basis

        pos1, pos2 are unused here since basis is precomputed for all psr pairs.
        they are here just to match the function signature of other orfs.

        Parameters
        ----------
        pos1 : float
            pulsar 1 position (unused)
        pos2 : float
            pulsar 2 position (unused)
        clm : array
            linear spherical harmonic coefficients
        """
        orf = jnp.tensordot(clm, linsph_basis, axes=1) # Npsr, Npsr
        return orf
    return linsph_orf



# Utility functions to keep doc comments when jitting
def jit_method(func):
    """A wrapper to JIT compile class methods with a static 'self' as first argument.

    This decorator takes a class method and returns a JIT compiled version of it
    while preserving the original function's metadata using functools.wraps.

    This is a drop-in replacement for: `@partial(jit, static_argnums=0)`

    Parameters
    ----------
    func : callable
        The function to be wrapped and JIT compiled.

    Returns
    -------
    callable
        The JIT compiled version of the input function.
    """
    from jax import jit
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        jit_func = jit(func, static_argnums=0) # self is static
        return jit_func(*args, **kwargs)
    
    return wrapper


def jit_function(func):
    """A wrapper to JIT compile standalone functions.

    This decorator takes a standalone function and returns a JIT compiled version
    of it while preserving the original function's metadata using functools.wraps.

    This is a drop-in replacement for: `@jit`

    Parameters
    ----------
    func : callable
        The function to be wrapped and JIT compiled.

    Returns
    -------
    callable
        The JIT compiled version of the input function.
    """
    from jax import jit
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        jit_func = jit(func)
        return jit_func(*args, **kwargs)
    
    return wrapper

