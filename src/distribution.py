import numpy as np
import pandas as pd
import scipy.stats


def fit_distribution(data, bins=100,dist='lognorm'):
    if dist=='lognorm':
        print('Fitting LogNorm')
        dist=scipy.stats.lognorm
    elif dist=='norm':
        print('Fitting Norm')
        dist=scipy.stats.norm


    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Estimate distribution parameters from data
    params = dist.fit(data)
    
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1] # STD

    # Calculate fitted PDF and error with fit in distribution
    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
    
    res = y - pdf
    sq_res = np.power(y - pdf, 2.0)
    ssr = np.sum(np.power(y - pdf, 2.0))

    pdf, x_axis = make_pdf(dist, params, x=None)

    return res,pdf,x,params


def flatten(weights, day=None):
    if day is None:
        flattened = np.ravel(weights)
    else:
        flattened = weights[:, day]

    nonzero = flattened[flattened > 0]

    return nonzero

def make_pdf(dist, params, size=10000, x=None):
    """Generate distributions's Probability Distribution Function """
    print('Generating {} with params {} of size {}'.format(dist,params,size))
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = (
        dist.ppf(0.01, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.01, loc=loc, scale=scale)
    )
    end = (
        dist.ppf(0.99, *arg, loc=loc, scale=scale)
        if arg
        else dist.ppf(0.99, loc=loc, scale=scale)
    )

    # Build PDF and turn into pandas Series
    if x is None:
        x = np.linspace(start, end, size)
    else:
        x = x

    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    

    return pdf,x
