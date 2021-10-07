import numpy as np
SPINE_TAG=2
SHAFT_TAG=1

def get_presence(data,split_unsures=False):
    syn_mtx = data.synapseMatrix
    binary_mtx = syn_mtx>0
    binary_mtx = binary_mtx.astype('int')

    return binary_mtx

def calc_survival_fraction(binary_mtx,day=0):
    originals = binary_mtx[binary_mtx[:,day]>0,:]
    sf = np.mean(originals,axis=0)

    if len(sf)<8: # for PV with only 7 days
        sf = np.append(sf,np.nan)

    return sf

def calc_norm_dens(binary_mtx):
    n_syn = np.sum(binary_mtx,axis=0)
    norm_dens = n_syn / n_syn[0]

    if norm_dens.shape[0] < 8:
        norm_dens = np.append(norm_dens,np.nan)
    return norm_dens


def get_distances(data):
        numDays = data.synapseMatrix.shape[1]
        distance_mat = np.empty((data.synapseMatrix.shape))
        distance_mat.fill(np.nan)
        all_synIDs = list(data.synapseID)

        for i in range(numDays):
            micron_dist = data.distances.clustData[i].micronsAlongDendrite
            curr_synIDs = data.distances.clustData[i].synIDs
            for curr_sidx, sid in enumerate(curr_synIDs):
                if str(sid) in all_synIDs:
                    s_idx = all_synIDs.index(str(sid))
                    distance_mat[s_idx, i] = micron_dist[curr_sidx]

        return distance_mat

def calc_true_dens(data):
    distance_matrix = get_distances(data)
    
    binary = get_presence(data)
    # print(binary)
    norm_dens = calc_norm_dens(binary)

    mins = np.nanmin(distance_matrix, axis=0)
    maxs = np.nanmax(distance_matrix, axis=0)
    dendrite_lengths = maxs - mins
    n_syn = np.sum(binary,axis=0)

    total_density = n_syn / dendrite_lengths

    synapse_matrix = data.synapseMatrix

    spines = synapse_matrix == SPINE_TAG
    shafts = synapse_matrix == SHAFT_TAG

    spines = spines.astype("int")
    shafts = shafts.astype("int")

    num_spines_byday = np.sum(spines, axis=0)
    num_shafts_byday = np.sum(shafts, axis=0)

    spine_density = num_spines_byday/dendrite_lengths
    shaft_density = num_shafts_byday/dendrite_lengths

    
    return (total_density,spine_density,shaft_density)

