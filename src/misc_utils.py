import numpy as np
import os
import fnmatch
import sys
from main_utils import Dendrite

def inverse_transform_sampling(empirical_data, num_samples, bins='auto', seed_val=0):
	# inspired by http://usmanwardag.github.io/python/astronomy/2016/07/10/inverse-transform-sampling-with-python.html
	# and slide 14 of http://www.cse.psu.edu/~rtc12/CSE586/lectures/cse586samplingPreMCMC.pdf
	if seed_val is not None:
		np.random.seed(seed_val)
	out_hist, out_bins = np.histogram(empirical_data, bins=bins)
	prob = out_hist / float(np.sum(out_hist))
	cum_prob = np.cumsum(prob)
	U = np.random.uniform(0, 1, num_samples)
	sampled_data = np.squeeze(np.array([out_bins[np.argwhere(cum_prob == min(cum_prob[(cum_prob - r) > 0]))] for r in U]))
	return sampled_data

def gather_binary(metric,data_path,split=False,hq=False,print_progress=True):
    metric_agg = []
   
    
    length_sevens = ['num_add','num_elim','norm_add','norm_elim','tor']
    length_eights = ['norm_dens','num_syn','sf','spine_density','shaft_density','total_density']
    
    if metric in length_sevens:
        metric_length = 7
    elif metric in length_eights:
        metric_length = 8
    else:
        print('Metric not in length_lists!')
        
        
    
    for dirpath,dirs,files in os.walk(data_path):
        for nm_f in fnmatch.filter(files, '*.mat'):
            if print_progress:
                print(nm_f)
                
            d = Dendrite(os.path.join(data_path, nm_f),fluorescence_kwargs={'norm_agg':'median'},split_unsures=split,high_quality_only=hq)
            
            metric_data = d.binary_dynamics[metric]
            
            if len(metric_data) is not metric_length:
                if print_progress:
                    print('LENGTH MISMATCH FOUND')
                    
                assert len(metric_data) == (metric_length - 1)
                if print_progress:
                    print('LENGTH MISMATCH RESOLVED: nan appended')
                metric_data = np.append(metric_data,np.nan)
            metric_agg.append(metric_data)                
            
            
    return np.asarray(metric_agg)

def sterror(data,axis=0):
    data = np.squeeze(data)
    num_dim = len(data.shape)
    
    # check input dimensionality
    if num_dim is 1:
        vector = True
        matrix = False
        
    elif num_dim is 2:
        matrix = True
        vector = False
    
    else:
        matrix = False
        vector = False
   
    # main calculations
    if vector:
        ste = np.nanstd(data) / np.sqrt(len(data))
        print('Calculated STE from a vector of shape {} and used an N of {}'.format(len(data),len(data)))
        
        return ste
        
    elif matrix:
        ste = np.nanstd(data,axis=axis) / np.sqrt(data.shape[axis])
        print('Calculated STE from matrix of shape {}, collapsed across dimension {}, and used an N of {}!'.format(data.shape,axis,data.shape[axis]))
        
        return ste
        
    elif not matrix and not vector:
        print('Improper input dimensionality. Data must be a vector (1xN) or a matrix (NxM)!')
        