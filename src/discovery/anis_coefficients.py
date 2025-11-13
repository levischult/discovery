import jax.numpy as jnp
import numpy as np
import healpy as hp
import scipy.special as ss
import discovery.deterministic as dsd
#from discovery.deterministic import fpc_fast
from time import time_ns
import jax
from jax.scipy import linalg as jsl


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




































######################

def spherical_to_cartesian(theta, phi):
    """
    Kyle Gersbach
    Convert unit-spherical coordinates to unit-cartesian coordinates.

    This function converts spherical (theta, phi) coordinates to cartesian (x, y, z).
    This is done by using the following transformations:
    - x = sin(theta) * cos(phi)
    - y = sin(theta) * sin(phi)
    - z = cos(theta)

    This function is vectorized to handle multiple angles at once.

    Args:
        theta (scalar, ndarray): The polar angle in radians.
        phi (scalar, ndarray): The azimuthal angle in radians.

    Returns:
        ndarray: The cartesian coordinates (x, y, z) with shape (3, n).
    """
    ret = np.array([np.sin(theta)*np.cos(phi),
                    np.sin(theta)*np.sin(phi),
                    np.cos(theta)])
    return ret # [3, n]

def psr_antenna_response(psr_theta, psr_phi, gw_theta, gw_phi, npix=None):
    """
    Kyle Gersbach
    Calculate the pulsar antenna response function for a given set of pulsar and GW directions.

    This function computes the pulsar antenna response function for a given set 
    of pulsar and GW directions for both the pulsar plus and cross polarizations.
    The response function is normalized by sqrt(3/(2*npix)) such that when creating
    correlation functions Gamma_{ab} = Fp[a]Fp[b] + Fc[a]Fc[b] it is normalized
    to have Gamma = 1/2 for 0 separation.

    npix is the total number of pixels in the sky map, which is used for normalization.
    If npix is None, it is set to the length of gw_theta, which is assumed to be 
    the total number of pixels.

    This function is vectorized to handle multiple pulsars and GW directions at once.

    Args:
        psr_theta (ndarray): The polar angles of the pulsars in radians.
        psr_phi (ndarray): The azimuthal angles of the pulsars in radians.
        gw_theta (ndarray): The polar angles of the gravitational wave source direction in radians.
        gw_phi (ndarray): The azimuthal angles of the gravitational wave source direction in radians.
        npix (int, optional): The total number of pixels in the sky map. Defaults to None.

    Returns:
        tuple: A tuple containing two ndarrays:
            - Fp (ndarray): Pulsar plus polarization response per pixel [npsr, npix]
            - Fc (ndarray): Pulsar cross polarization response per pixel [npsr, npix]
    """
    npix = len(gw_theta) if npix is None else npix

    # Unit vector in direction of pulsar
    phat = spherical_to_cartesian(psr_theta, psr_phi) # [3, npsr]
    # Unit vector in direction of GW
    omegahat = spherical_to_cartesian(gw_theta, gw_phi) # [3, npix]
    # Gair et al 2014 use propagation direction, not the source direction
    omegahat = -omegahat # [3, npix]

    # For polarization basis tensor
    mhat = np.array([-np.sin(gw_phi), 
                    np.cos(gw_phi), 
                    np.zeros_like(gw_phi)]) # [3, npix]
    nhat = np.array([-(np.cos(gw_theta)*np.cos(gw_phi)), 
                    -(np.cos(gw_theta)*np.sin(gw_phi)), 
                    (np.sin(gw_theta))]) # [3, npix]

    # p indexes pulsars, n indexes pixels, i indexes components
    # Dot products using einsum 'in,ip->pn' or using matrix multiplication (faster)
    omega_dot_p = phat.T @ omegahat # [npsr, npix]
    m_dot_p = phat.T @ mhat # [npsr, npix]
    n_dot_p = phat.T @ nhat # [npsr, npix]

    # Plus polarization
    Fp = (0.5)*(n_dot_p**2 - m_dot_p**2) / (1 - omega_dot_p) # [npsr, npix]
    # Cross polarization
    Fc = (m_dot_p * n_dot_p) / (1 - omega_dot_p) # [npsr, npix]

    # Normalize by sqrt(3/(2*npix)) to ensure that the response function has correlations
    norm = np.sqrt(3/(2*npix))    

    return Fp*norm, Fc*norm # [npsr, npix]

def createSignalResponse(psrs, nside):
    """
    Kyle Gersbach, Levi Schult
    Constructs the Signal Response matrix in the pixel basis

    This function creates the Signal Response matrix for every pixel on the sky.
    psrs is assumed to be a list of Pulsar objects (has .theta .phi attributes)
    and nside for healpy resolution.

    This includes effects of pulsar term by doubling auto correlations (diagonal)
    of ORF matrices.

    Parameters
    ----------
    psrs : list of Pulsar objects
        pulsar positions used to construct ORFs
    nside : _type_
        healpy resolution for constructing ORF/pixel on sky (pixel basis)

    Returns
    -------
    array
        npix, npsr, npsr array. [0,:,:] is npsr, npsr ORF for 0th pixel in sky.
    """
    psrs_theta = jnp.array([p.theta for p in psrs]) # npsrs
    psrs_phi = jnp.array([psr.phi for psr in psrs]) # npsrs
    npsrs = len(psrs)
    npix = hp.nside2npix(nside)
    gw_theta, gw_phi = hp.pix2ang(nside=nside, ipix=np.arange(hp.nside2npix(nside))) # npix, npix

    # ORF is in terms of GW source direction
    Fp, Fc = psr_antenna_response(psrs_theta, psrs_phi, gw_theta=gw_theta, gw_phi=gw_phi) # npsr x npix each
    a, b = jnp.triu_indices(len(psrs), k=0) # npairs, npairs includes auto-pairs
    R_ab = Fp[a]*Fp[b] + Fc[a]*Fc[b] # npairs x npix
    R = jnp.zeros((npsrs, npsrs, npix))
    R = R.at[a, b].set(R_ab) # unpacking npairs x npix into npsr x npsr array (upper triangle)
    R = R.at[b, a].set(R_ab) # lower triangle now - ignoring npix axis because R[a, b] is calling npairs x npix
    # include pulsar term on diagonal == double the diagonal manually
    idx = jnp.arange(npsrs)
    R = R.at[idx, idx].set(R[idx, idx]*2)
    R = jnp.swapaxes(R, axis1=0, axis2=2) # have npix axis first
    return R

def createRadiometerResponse(psrs, nside):
    """
    Levi Schult, Kyle Gersbach
    _summary_

    Parameters
    ----------
    psrs : _type_
        _description_
    nside : _type_
        _description_
    """
    npix = hp.nside2npix(nside)
    Gam = createSignalResponse(psrs, nside)
    Gam *= npix # multiply by npix (one pixel at a time has all sky power)
    # for normalization, that one pixel has power = npix
    # must iterate over Gamma's pixels
    return Gam


#### SPHERICAL HARMONICS ####

def real_sph_harm(mm, ll, phi, theta):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF
    The real-valued spherical harmonics.
    """
    if mm > 0:
        ans = (1.0 / np.sqrt(2)) * (ss.sph_harm(mm, ll, phi, theta) + ((-1) ** mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm == 0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm < 0:
        ans = (1.0 / (np.sqrt(2) * complex(0.0, 1))) * (
            ss.sph_harm(-mm, ll, phi, theta) - ((-1) ** mm) * ss.sph_harm(mm, ll, phi, theta)
        )

    return ans.real

def clmFromAlm(alm):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
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

def almFromClm(clm):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function
    just takes the imaginary part of the abs(m) alm index.
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

def mapFromClm(clm, nside):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF: mapFromClm_fast

    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm))) - 1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h

def clmFromMap(h, lmax):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF: clmFromMap_fast
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values

    Use Healpix spherical harmonics for computational efficiency
    """
    alm = hp.sphtfunc.map2alm(h, lmax=lmax)
    alm[0] = np.sum(h) * np.sqrt(4 * np.pi) / len(h)

    return clmFromAlm(alm)


# LSS This will need testing since it takes in a 
# signal response matrix from createSignalResponse_fast
def getCov(clm, nside, F_e):
    """
    DIRECT IMPORT FROM ENTERPRISE ANIS_COEF
    Given a vector of clm values, construct the covariance matrix

    @param clm:     Array with Clm values
    @param nside:   Healpix nside resolution
    @param F_e:     Signal response matrix

    @return:    Cross-pulsar correlation for this array of clm values
    """
    # Create a sky-map (power)
    # Use mapFromClm to compare to real_sph_harm. Fast uses Healpix
    # sh00 = mapFromClm(clm, nside)
    sh00 = mapFromClm(clm, nside)

    # Double the power (one for each polarization)
    sh = np.array([sh00, sh00]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))