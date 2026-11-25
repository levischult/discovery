import jax.numpy as jnp
import numpy as np
import healpy as hp
import pickle as pkl

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

    Mprime = jnp.zeros((npsr, npsr, nbasis))
    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = Mprime.at[a, b, :].set(M)
    Mprime = Mprime.at[b, a, :].set(M)

    return Mprime


def matrix2pairs(M):
    # M has shape (npsr, npsr, N) -> return Mp with (npairs, N)
    npsr = M.shape[1]
    nbasis = M.shape[-1]
    npair = npsr*(npsr+1)//2

    Mprime = jnp.zeros((npair, nbasis))
    a,b = jnp.triu_indices(npsr, k=0)
    Mprime = Mprime.at[:, :].set(M[a, b, :])

    return Mprime

def plot_pulsars(dataset, highlight=None, returnphitheta=False):
    '''
    plots pulsar positions from a dataset pkl 
    if you want to save the pulsar positions, set returnphitheta=True
    
    inputs:
    dataset (str): path to dataset
    highlight (list): list of pulsar names as strings

    outputs:
    psrphis (array): phi of pulsars
    psrthetas (array): theta of pulsars

    usage:
    psrphis, psrthetas = pulsarphitheta('location/of/dataset.pkl')
    '''
    # bringing in pulsars 
    try:
        with open(dataset , 'rb') as ff:
            psrs1 = pkl.load(ff)
    except TypeError:
        psrs1 = dataset # LSS if the dataset is already a list of pulsars, just use it directly
    # LSS getting psr locations
    psrphis = np.array([psr.phi for psr in psrs1])
    psrthts = np.array([psr.theta for psr in psrs1])
    # LSS plotting
    hp.visufunc.projscatter(psrthts, psrphis,
                    marker='*',color='white',
                    edgecolors='k',s=100)
    if highlight is not None:
        psrnames = [p.name for p in psrs1] # LSS highlight pulsars based on name
        highlight_idx = []
        for n in highlight:
            highlight_idx.append(psrnames.index(n))
        hp.visufunc.projscatter(np.array(psrthts)[highlight_idx], 
                                np.array(psrphis)[highlight_idx],
                    marker='*',color='yellow',
                    edgecolors='k',s=100)
    if returnphitheta:
        return psrphis, psrthts

def chain_skymap(phi, theta, nside, ct=False):
    '''
    makes an array of pixel values for healpy plotting from sky location posteriors

    inputs:
    phi (array/list): burned chain of phi samples from mcmc analysis
    costheta (array/list): burned chain of costheta samples from mcmc analysis
    nside (int): should be power of 2 - the resolution of output healpy map (12Nside**2)
    ct (bool): default false - if you want to pass in costheta directly, set to True.

    outputs:
    skymap - an array of pixel values to be passed to hp.mollview or other proj.
    '''
    npix = hp.nside2npix(nside)

    # convert to HEALPix indices
    if ct:
        indices = hp.ang2pix(nside, np.arccos(theta), phi)
    else: indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=float)
    normcts = counts / np.sum(counts) # LSS normalizing
    hpx_map[idx] = normcts

    return hpx_map