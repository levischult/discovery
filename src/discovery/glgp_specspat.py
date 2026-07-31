import discovery as ds
import jax.numpy as jnp
import numpy as np
import jax
import re
import inspect
import typing

fourierbasis = ds.fourierbasis
matrix = ds.matrix

# LSS here we are going to make a custom global gp that can take in multiple
# spatial and spectral models. N spec = N spatial models. Included is the ability 
# to have custom spatial models that vary at each freq or are different at each freq.
# Additionally the spatial parameters can be parametrized with parameters searched over.
# For right now, nfreqs is equivalent across spectral models.


# LSS important usage items
# priors should always be a list
# orfs should always be a list of lists, even if you are only 
# doing one spatial model. e.g. [[hd_orf]] not [hd_orf]
# nbasis should always be a list, even if doing one spatial model.
# if you are doing more than one spatial model, then think carefully about how
# the spatial model and spectral model match



def makeglobalgp_fourier_specspat(psrs, priors, orfs, components, T, fourierbasis=fourierbasis,
                                  exclude=['f', 'df', 'pos1', 'pos2'], name='fourierGlobalGP', 
                                  nbasis=None, orfnames=None, fast=True):
# def makeglobalgp_fourier(psrs, priors, orfs, components, T, fourierbasis=fourierbasis, means=None, common=[], exclude=['f', 'df'],
#                          name='fourierGlobalGP', meansname='meanFourierGlobalGP')
    priors = priors if isinstance(priors, list) else [priors]
    orfs   = orfs   if isinstance(orfs, list)   else [orfs]
    # LSS in this, we want each prior to have a set of ORFs for each frequency.
    # LSS if you give a prior that is broadband in prescription, then it should
    # use the same orf for all frequencies.

    # LSS DO NOT PUT IN MORE THAN ONE SPECTRAL MODEL I.E. len(priors) == 1 !!!!



    if len(priors) != len(orfs):
        raise ValueError("I need as many priors as ORFs.")

    # LSS check that number of orfs equal number components
    # print(priors)
    # argspec = inspect.getfullargspec(prior[0])
    # for arg in argspec.args:
    #     if argspec.annotations.get(arg) == typing.Sequence:
    #         if len(orfs) != components:
    #             raise ValueError('I need as many orfs as freqs if doing narrowband analysis')
    

    argmaps, orfmaps = [], [] # LSS again we want a list of lists for orfmaps to keep track of which spatial model matches the spectral model.
    prinames = []
    for pridx, prior in enumerate(priors):
        if len(priors) == 1: 
            priorname = f'{name}'
        else:
            if orfnames is None:
                raise ValueError('I need orfnames to name the priors if doing more than one spectral model')
            # LSS this needs to be fixed - at present it can't handle naming spectral args with orf name because there
            # is maybe too many different options for ORFS to be paired with a spectral model to assoc the orf with specral args
            # additionally, one often jits the spatial model removing the name attribute
            #priorname = f'{name}_{re.sub("_", "", modnames[pridx][0])}'
            priorname = f'{name}_{re.sub("_", "", prior.__name__)}_{re.sub("_", "", orfnames[pridx][0])}'

        prinames.append(priorname)
        argspec = inspect.getfullargspec(prior)
        argmaps.append([f'{priorname}_{arg}' + (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
                        for arg in argspec.args if arg not in exclude])
    
    
    # LSS Here is where we handle arguments for the ORFs if they are parametrized.
    
    for pridx, orf in enumerate(orfs):
        freqiter = 0 # LSS dummy iterator to keep track of which frequency each ORF corresponds to in varied basis per frequency case.
        priorname = prinames[pridx] # LSS iterate over prior names to name the ORF parameters.
        # LSS handling if you supply 1 ORF for a spectral model.
        
        # LSS this list will hold all the argmaps for a spatial model so our orfmaps list has len=priors
        orfmp = [] 
        
        if len(orf) == 1:
            orfspec = inspect.getfullargspec(orf[0])
            for arg in orfspec.args:
                if arg not in exclude:
                    if orfspec.annotations.get(arg) == typing.Sequence:
                        orfmp.append([f'{priorname}_{arg}' + f'({nbasis[pridx][0]})'])
                    else:
                        orfmp.append([f'{priorname}_{arg}'])
            orfmaps.append(orfmp)


        # LSS here we add orf arguments for each frequency if we are doing a narrowband search 
        else:
            orfspec = []
            for orfunc in orf: # LSS dealing with ORF for each freq of spec model.
                orfspec.append(inspect.getfullargspec(orfunc))
            for orfiter, orfspec_i in enumerate(orfspec): # LSS i here is iterating over the ORF assigned to each freq.
                for arg in orfspec_i.args:
                    if arg not in exclude:
                        if orfspec_i.annotations.get(arg) == typing.Sequence:
                            orfmp.append([f'{priorname}_{arg}_f{freqiter}' + f'({nbasis[pridx][orfiter]})'])
                        else:
                            orfmp.append([f'{priorname}_{arg}_f{freqiter}'])
                freqiter += 1
            orfmaps.append(orfmp)

    fs, dfs, fmats = zip(*[fourierbasis(psr, components, T) for psr in psrs])
    f, df = matrix.jnparray(fs[0]), matrix.jnparray(dfs[0])
    psrpos = [matrix.jnparray(psr.pos) for psr in psrs]


    def priorfunc(params):
        # LSS phis is the len=2*nfreq PSD
        # phis = [prior(f, df, *[params[arg] for arg in argmap]) for prior, argmap in zip(priors, argmaps)]
        # print(f'{params=}')
        phis = [] # LSS list of len=2*nfreq PSDs for each spec model.
        for pridx, prior in enumerate(priors):
            argmap = argmaps[pridx]
            # print(f'{argmap=}')
            # print(f'{prior=}')
            # print(f'{prior(f, df, *[params[arg] for arg in argmap])=}')
            # print(f'{[params[arg] for arg in argmap]=}')
            # print(f'{params['gw_log10_rho(3)']=}')
            phis.append(prior(f, df, *[params[arg] for arg in argmap]))
        
        # print(f'{phis=}')
        # print(f'{phis[0].shape=}')

        # LSS orf function here returns the entire npsr, npsr matrix
        allorfmats = [] 
        # LSS list of orfmats that are nfreq x npsr x npsr shape for 
        # each spatial model that corresponds to a spectral model.
        # if there is only one spatial model, 
        # then we duplicate it across frequencies to achieve standard shape
        for pridx, orf in enumerate(orfs):
            orfmap = orfmaps[pridx]
            if len(orf) == 1:
                # LSS just get the argmap for the one ORF since a single spatial model 
                if len(orfmap) > 0: # LSS handling fixed ORF case.
                    orfmap = orfmaps[pridx][0] 
                singlefreq_orf = matrix.jnparray(orf[0](None, None, *[params[arg] for arg in orfmap]))
                # LSS if one ORF for all freqs, duplicate to get nfreq, npsr, npsr shape
                # LSS make it 1, npsr, npsr shape so we can repeat it.
                singlefreq_orf = jnp.array([singlefreq_orf]) 
                orfmats = np.repeat(singlefreq_orf, components, axis=0)
                allorfmats.append(orfmats)
            else: # LSS this is handling a freq specific ORF where we have an ORF for each frequency of a spectral model. 
                # LSS In this case, we just iterate over the ORFs and get the orfmat for each freq and stack them to get nfreq, npsr, npsr shape.
                orfholder = []
                for orf_i, orfmap_i in zip(orf, orfmap):
                    orfholder.append(matrix.jnparray(orf_i(None, None, *[params[arg] for arg in orfmap_i])))
                orfmats = jnp.array(orfholder)
                allorfmats.append(orfmats)

        # LSS make a 2nfreq, npsr, npsr matrix out of our ORFs
        bigorfmats = []
        for orfmats in allorfmats:
            bigorfmat = jnp.block(np.repeat(orfmats, 2, axis=0)) # LSS should be 2nfreq, npsr, npsr
            bigorfmats.append(bigorfmat)

        # LSS this is compressed phi: shape is 2nfreq, npsr, npsr
        Phicomps = []
        for pridx, bigorfmat in enumerate(bigorfmats):
            # print(phis)
            # print(f'{phis[pridx][:, jnp.newaxis, jnp.newaxis].shape=}')
            # print(f'{bigorfmat.shape=}')
            # print(f'{phis[pridx][:, jnp.newaxis, jnp.newaxis]=}')
            # print(f'{bigorfmat}')
            Phicomp = phis[pridx][:, None, None] * bigorfmat
            # print(f'{Phicomp.shape=}')
            # print(f'{Phicomp=}')
            Phicomps.append(Phicomp)

        # LSS now we need to sum over spectral/spatial model combinations to get
        # the full compressed Phi matrix that is 2nfreq, npsr, npsr shape
        Phicomp = jnp.sum(jnp.array(Phicomps), axis=0)

        #print(Phicomp.shape)

        # LSS unpack Phi now - this is how michele did it in multiorf code.
        n_diag, n_row_blocks, n_col_blocks = Phicomp.shape
        blocks = jnp.zeros((n_row_blocks, n_col_blocks, n_diag, n_diag))
        blocks = blocks.at[..., jnp.arange(n_diag), jnp.arange(n_diag)].set(Phicomp.transpose(1, 2, 0))
        Phi = blocks.transpose(0, 2, 1, 3).reshape(n_row_blocks*n_diag, n_col_blocks*n_diag)
        return Phi
    
    # LSS composing all the params
    priset = []
    for argmap in argmaps:
        priset.append(set(argmap))

    orfset = []
    for orfmap in orfmaps:
        for arg in orfmap:
            orfset.append(set(arg))
    
    fullset = set.union(*priset, *orfset)
    priorfunc.params = sorted(fullset)
    priorfunc.type = jax.Array

    def invprior(params):
        Phi = priorfunc(params)
        return jnp.linalg.inv(Phi), jnp.linalg.slogdet(Phi)[1]
    
    invprior.params = priorfunc.params
    invprior.type = jax.Array

    gp = matrix.GlobalVariableGP(matrix.NoiseMatrix12D_var(priorfunc), fmats)
    gp.Phi_inv, gp.factors = invprior, None

    gp.index = {f'{psr.name}_{name}_coefficients({len(f)})':
                slice(len(f)*i, len(f)*(i+1)) for i, psr in enumerate(psrs)}
    gp.pos = [psr.pos for psr in psrs]
    gp.name = [psr.name for psr in psrs]

    return gp

# we'll have a powerlaw HD process...
def hd_orf(pos1, pos2):
    x = jnp.dot(pos1, pos2)
    omc2 = (1.0 - x) / 2.0
    val = 1.5 * omc2 * jnp.log(omc2 + 1e-12) - 0.25 * omc2 + 0.5
    return jnp.where(jnp.abs(x - 1.0) < 1e-6, 1.0, val)

# ...and another powerlaw with a custom ORF; here you'd do your pixel ORF
def myorf(pos1, pos2, a):
    return a * jnp.dot(pos1, pos2)

# and then you'd set up your likelihood with, say,
# globalgp = makeglobalgp_fourier_multiorf(psrs, [ds.powerlaw, ds.powerlaw], [hd_orf, myorf], components=14, T=T, name='gw')
