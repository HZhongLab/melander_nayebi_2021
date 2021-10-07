import numpy as np
import itertools
import copy
import collections

def return_criterion_func(criterion='dynamic'):
    if criterion == 'dynamic':
        criterion_func = lambda x: x != 0
    elif criterion == 'addition':
        criterion_func = lambda x: x == 1
    elif (criterion == 'deletion') or (criterion == 'elimination'):
        criterion_func = lambda x: x == -1
    else:
        raise ValueError
    return criterion_func
    
def create_dynamic_events_matrix(synapse_matrix):
    M = np.array(synapse_matrix)
    B = M > 0
    P = B.astype('float64')
    D = []
    for i in range(P.shape[1]-1):
        D.append(np.expand_dims(P[:, i+1] - P[:, i], axis=-1))
    D = np.concatenate(D, axis=-1) # create dynamic events matrix
    return D

def create_persistent_synapse_matrix(synapse_matrix):
    M = np.array(synapse_matrix)
    B = M > 0
    P = B.astype('float64')
    D = np.logical_and.reduce(P.astype('bool'), axis=-1).astype('float32')
    return D

def get_persistent_distances(dend):
    pers_D = dend.persistent_matrix
    persistent_distances = np.median(dend.rescaled_distance[np.where(pers_D == 1)], axis=1)
    return persistent_distances

def simulate_random_dendrite_distances(dend, sim_size=100, low_val=0, seed_val=0):
    if seed_val is not None:
        np.random.seed(seed_val)
    persistent_distances = get_persistent_distances(dend)
    maxPersistentDistance = np.amax(persistent_distances)
    dynamic_distances_arr = np.random.uniform(low=low_val, high=maxPersistentDistance, size=sim_size)
    return dynamic_distances_arr

def get_dynamic_synapse_dists(dend, criterion='dynamic', cluster_per_day=True, random=False, seed_val=0):
    if seed_val is not None:
        np.random.seed(seed_val)
    D = dend.dynamics_matrix
    rescaledDist = dend.rescaled_distance
    if isinstance(criterion, str):
        criterion_func = return_criterion_func(criterion=criterion)
    else:
        criterion_func = None

    if criterion_func is not None:
        where_dynamic = np.where(criterion_func(D))
    elif not isinstance(criterion, str): # criterion is a list of [[synapse], [day]] pairs
        where_dynamic = criterion

    assert(len(where_dynamic) == 2)
    synapse_indices = where_dynamic[0]
    day_indices = where_dynamic[1]
    num_dynamic_syn = len(synapse_indices)
    # we want to count each dynamic synapse only once if we are NOT clustering per day
    if (num_dynamic_syn > 0) and (not cluster_per_day):
        repeated_synapses = [item for item, count in collections.Counter(synapse_indices).items() if count > 1]
        if len(repeated_synapses) > 0: # exists synapse that was both added and eliminated within a month and our criterion was 'dynamic'
            indices_toremove = []
            for rs in repeated_synapses:
                rs_indices = np.array([i for i, x in enumerate(synapse_indices) if x == rs])
                curr_delete_indices = rs_indices[np.argsort(day_indices[rs_indices])[:-1]] # keep only last day of dynamic synapse
                indices_toremove.extend(curr_delete_indices)
            synapse_indices = np.delete(synapse_indices, indices_toremove)
            day_indices = np.delete(day_indices, indices_toremove)

    dynamic_distances = {}
    for sd in zip(synapse_indices, day_indices):
        curr_s = sd[0]
        curr_d = sd[1]
        if D[sd] == 1: # addition on day d + 1
            if random:
                distance_val = np.random.uniform(low=np.nanmin(rescaledDist[:, curr_d+1]), 
                                                 high=np.nanmax(rescaledDist[:, curr_d+1]), size=1)[0]
            else:
                distance_val = rescaledDist[curr_s, curr_d+1]
        elif D[sd] == -1: # deletion on day d + 1
            if random:
                distance_val = np.random.uniform(low=np.nanmin(rescaledDist[:, curr_d]), 
                                                high=np.nanmax(rescaledDist[:, curr_d]), size=1)[0]
            else:
                distance_val = rescaledDist[curr_s, curr_d]
        else:
            raise ValueError
        dynamic_distances[sd] = distance_val
    return dynamic_distances

def find_nearest_persistent_synapse(dynamic_distances, persistent_distances):
    nearest_distance = []
    if isinstance(dynamic_distances, dict):
        dd_arr = dynamic_distances.keys()
    else:
        dd_arr = range(len(dynamic_distances))
    for s in dd_arr:
        p_dists = [np.abs(p - dynamic_distances[s]) for p in persistent_distances]
        nearest_distance.append(np.amin(p_dists))
    return nearest_distance
    
def normalize_synapse_distances(distanceMat, N_prior=10, resetminval=0.0):
    '''Take average of N ~ 10 persistent synapses on first day 
    and then fit single scalar from the remaining days based on average pairwise distances'''
    synapseNan = np.isnan(distanceMat).sum(axis=1)
    minnanval = np.amin(synapseNan)
    mostPersistentSynapses = np.where(synapseNan == minnanval)[0]
    N_choose = min(mostPersistentSynapses.shape[0], N_prior)
    # TO DO: generalize this to the fluorescences where we exclude weak and too strong synapses, 
    # for now we pick 10 random synapses
    inds = np.random.RandomState(seed=0).permutation(mostPersistentSynapses.shape[0])[:N_choose]
    chosenSynapses = mostPersistentSynapses[inds]
    dist = []
    for p in list(itertools.combinations(chosenSynapses, 2)):
        dist.append(np.expand_dims(np.abs(distanceMat[p[0], :] - distanceMat[p[1], :]), axis=0))
    dist = np.concatenate(dist, axis=0) # synapse pairs x days matrix
    avgPairwiseDist = np.mean(dist, axis=0) # length days vector
    scaling_factor = (avgPairwiseDist[0])/(avgPairwiseDist) # avg pairwise distance at day 0 / avg pairwise distance per day
    scaling_factor = np.tile(np.expand_dims(scaling_factor, axis=0), [distanceMat.shape[0], 1])
    rescaledDistance = (scaling_factor)*(distanceMat)
    # subtract average rescaled distance of synapses per day
    avgdistance = np.mean(rescaledDistance[chosenSynapses, :], axis=0)
    rescaledDistance = rescaledDistance - avgdistance
    minRescaledDist = np.nanmin(rescaledDistance)
    if minRescaledDist < 0.0: # to distances make nonnegative 
        rescaledDistance = rescaledDistance - minRescaledDist
        # set minimum distance to be a defined minimum distance (default 0) along the dendrite
        rescaledDistance = rescaledDistance + resetminval
    return rescaledDistance
