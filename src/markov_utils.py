import numpy as np
import copy
import os
import fnmatch
import scipy
import scipy.io as spio
from scipy.stats import sem
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools

def preproc_intensity(intensity_mat):
    if len(np.where(intensity_mat == 0.0)[0]) > 0 and len(intensity_mat[np.isnan(intensity_mat)]) > 0:
        print('CAUTION: there exist synapses with 0 weight which will be simply marked as not present.')

    if len(intensity_mat[np.isnan(intensity_mat)]) > 0:
        intensity_mat = np.nan_to_num(intensity_mat, copy=True)

    day_sum = np.sum(intensity_mat, axis=1)
    intensity_mat = np.delete(intensity_mat, np.where(day_sum == 0)[0], axis=0) # to deal with all nan bug

    return intensity_mat

def get_stationary_arr(stationary_dist):
    if isinstance(stationary_dist, dict):
        stationary_dist_arr = np.array([v for k,v in stationary_dist.items()])
    else:
        stationary_dist_arr = stationary_dist
    return stationary_dist_arr

def bin_clip_func(orig_markov_bins, bin_clip_idx):
    new_markov_bins = np.delete(orig_markov_bins, range(bin_clip_idx, len(orig_markov_bins)), axis=0)
    new_markov_bins = np.append(new_markov_bins, orig_markov_bins[-1])
    return new_markov_bins

def bin_data(intensity_mat,
             bins_touse='auto',
             plot_hist=False,
             only_nonzero=True,
             bin_per_day=True):
    '''Returns a days x num_bins matrix'''
    intensity_mat = preproc_intensity(intensity_mat)
    if bin_per_day:
        all_bins = []
        for d in range(intensity_mat.shape[1]):
            if only_nonzero:
                nonz_idx = np.where(intensity_mat[:, d] > 0)[0]
                m_d = intensity_mat[nonz_idx, d]
            else:
                m_d = intensity_mat[:, d]

            n, bins, patches = plt.hist(m_d, bins=bins_touse)
            all_bins.append(bins)
            if plot_hist:
                if only_nonzero:
                    plt.title('Histogram of nonzero synaptic weight on day ' + str(d))
                else:
                    plt.title('Histogram of synaptic weight on day ' + str(d))
                plt.show()
        all_bins = np.array(all_bins)
    else:
        if only_nonzero:
            nonz_idx = np.where(intensity_mat > 0)
            m_z = intensity_mat[nonz_idx]
        else:
            m_z = intensity_mat

        n, all_bins, patches = plt.hist(m_z, bins=bins_touse)
    plt.clf()
    return all_bins

def compute_stationary_dist(intensity_mat, bins_touse, preproc=True):
    if preproc:
        intensity_mat = preproc_intensity(intensity_mat)
    stationary_distribution = OrderedDict()

    bin_counts_mat = []
    for idx in range(len(bins_touse)):
        if idx == 0:
            bin_left = 0
            bin_right = bins_touse[0]
        else:
            bin_left = bins_touse[idx-1]
            bin_right = bins_touse[idx]

        num_bin_vals = []
        for d in range(intensity_mat.shape[1]):
            day_intensity = intensity_mat[:, d]

            bin_idxs = np.where((bin_left <= day_intensity) & (day_intensity < bin_right))[0]

            num_bin_vals.append(len(bin_idxs))

        num_bin_vals = np.array(num_bin_vals)
        bin_counts_mat.append(num_bin_vals)
        normalized_bin_vals = num_bin_vals / (float)(intensity_mat.shape[0])
        stationary_distribution[(bin_left, bin_right)] = np.mean(normalized_bin_vals)
    bin_counts_mat = np.array(bin_counts_mat)
    return stationary_distribution, bin_counts_mat

def compute_conditional_dist(intensity_mat, stationary_dist, exclude_00=False, reverse=False, delta_time=1):
    intensity_mat = preproc_intensity(intensity_mat)
    states = stationary_dist.keys()
    distribution_counts = np.zeros((len(states), len(states)))
    denominator = 0.0
    for d1, d2 in zip(range(intensity_mat.shape[1]), range(intensity_mat.shape[1])[delta_time:]):
        if reverse:
            prev_intensity = intensity_mat[:, d2]
            curr_intensity = intensity_mat[:, d1]
        else:
            prev_intensity = intensity_mat[:, d1]
            curr_intensity = intensity_mat[:, d2]

        transition_mat = np.zeros((len(states), len(states)))
        for i1, s1 in enumerate(states):
            for i2, s2 in enumerate(states):
                s1_idx = np.where((s1[0] <= prev_intensity) & (prev_intensity < s1[1]))[0]
                s2_idx = np.where((s2[0] <= curr_intensity) & (curr_intensity < s2[1]))[0]
                overlap_s1s2 = set(s1_idx) & set(s2_idx)
                transition_mat[i1, i2] = len(list(overlap_s1s2))
        distribution_counts += transition_mat
        denominator += 1.0

    avg_dist_counts = distribution_counts / denominator # average counts
    if exclude_00: # exclude 0 to 0 transitions and normalize
        conditional_distribution = np.zeros(avg_dist_counts.shape)
        for i in range(len(states)):
            if i == 0:
                conditional_distribution[0, :] = avg_dist_counts[0, :] / np.sum(avg_dist_counts[0, 1:])
                conditional_distribution[0, 0] = 0.0
            else:
                conditional_distribution[i, :] = avg_dist_counts[i, :] / np.sum(avg_dist_counts[i, :])

    else:
        conditional_distribution = np.divide(avg_dist_counts, np.sum(avg_dist_counts, axis=1, keepdims=True))
    return conditional_distribution

def compute_stationary_dist_joint(intensity_mat, bins_touse):
    intensity_mat = preproc_intensity(intensity_mat)
    stationary_distribution = OrderedDict()

    bin_counts_mat = []
    for idx_1 in range(len(bins_touse)):
        for idx_2 in range(len(bins_touse)):
            if idx_1 == 0:
                bin_left_1 = 0
                bin_right_1 = bins_touse[0]
            else:
                bin_left_1 = bins_touse[idx_1-1]
                bin_right_1 = bins_touse[idx_1]

            if idx_2 == 0:
                bin_left_2 = 0
                bin_right_2 = bins_touse[0]
            else:
                bin_left_2 = bins_touse[idx_2-1]
                bin_right_2 = bins_touse[idx_2]

            num_bin_vals = []
            for d in range(intensity_mat.shape[1]):
                day_intensity = intensity_mat[:, d]

                bin_idxs_1 = np.where((bin_left_1 <= day_intensity) & (day_intensity < bin_right_1))[0]
                bin_idxs_2 = np.where((bin_left_2 <= day_intensity) & (day_intensity < bin_right_2))[0]

                num_bin_vals.append(len(bin_idxs_1) + len(bin_idxs_2))

            num_bin_vals = np.array(num_bin_vals)
            bin_counts_mat.append(num_bin_vals)
            normalized_bin_vals = num_bin_vals / (float)(intensity_mat.shape[0]*intensity_mat.shape[0])
            stationary_distribution[(bin_left_1, bin_right_1, bin_left_2, bin_right_2)] = np.mean(normalized_bin_vals)
    bin_counts_mat = np.array(bin_counts_mat)
    return stationary_distribution, bin_counts_mat

def compute_conditional_dist_joint(intensity_mat, stationary_dist):
    intensity_mat = preproc_intensity(intensity_mat)
    states = stationary_dist.keys()
    distribution_counts = np.zeros((len(states), len(states)))
    denominator = 0.0
    for d1, d2 in zip(range(intensity_mat.shape[1]), range(intensity_mat.shape[1])[1:]):
        prev_intensity = intensity_mat[:, d1]
        curr_intensity = intensity_mat[:, d2]

        transition_mat = np.zeros((len(states), len(states)))
        for i1, s1 in enumerate(states):
            for i2, s2 in enumerate(states):
                s1_idx_1 = np.where((s1[0] <= prev_intensity) & (prev_intensity < s1[1]))[0]
                s1_idx_2 = np.where((s1[2] <= prev_intensity) & (prev_intensity < s1[3]))[0]
                s2_idx_1 = np.where((s2[0] <= curr_intensity) & (curr_intensity < s2[1]))[0]
                s2_idx_2 = np.where((s2[2] <= curr_intensity) & (curr_intensity < s2[3]))[0]

                overlap_s1s2 = []

                for s21_i in s2_idx_1:
                    if s21_i in s1_idx_1:
                        overlap_s1s2.append(s21_i)

                    if s21_i in s1_idx_2:
                        overlap_s1s2.append(s21_i)

                for s22_i in s2_idx_2:
                    if s22_i in s1_idx_1:
                        overlap_s1s2.append(s22_i)

                    if s22_i in s1_idx_2:
                        overlap_s1s2.append(s22_i)

                transition_mat[i1, i2] = len(overlap_s1s2)
        distribution_counts += transition_mat
        denominator += 1.0

    avg_dist_counts = distribution_counts / denominator # average counts
    conditional_distribution = np.divide(avg_dist_counts, np.sum(avg_dist_counts, axis=1, keepdims=True))
    return conditional_distribution, distribution_counts

def verify_detailed_balance(conditional_mat, stationary_dist,exclude_adjacent=0):

    stationary_dist_arr = get_stationary_arr(stationary_dist)
    p_jip_i = []
    p_ijp_j = []
    for c in list(itertools.combinations(range(len(stationary_dist_arr)), 2)):
        i_idx = c[0]
        j_idx = c[1]
        if np.abs(i_idx - j_idx) > exclude_adjacent:
            p_jip_i.append(conditional_mat[j_idx, i_idx]*stationary_dist_arr[i_idx])
            p_ijp_j.append(conditional_mat[i_idx, j_idx]*stationary_dist_arr[j_idx])
    p_jip_i = np.array(p_jip_i)
    p_ijp_j = np.array(p_ijp_j)
    return p_ijp_j, p_jip_i

def plot_conditionals_together(conditional_mat, stationary_dist):
    stationary_dist_arr = get_stationary_arr(stationary_dist)
    p_ji = []
    p_ij = []
    for c in list(itertools.combinations(range(len(stationary_dist_arr)), 2)):
        i_idx = c[0]
        j_idx = c[1]
        p_ji.append(conditional_mat[j_idx, i_idx])
        p_ij.append(conditional_mat[i_idx, j_idx])
    p_ji = np.array(p_ji)
    p_ij = np.array(p_ij)
    return p_ij, p_ji

def compute_conditional_mean_var(conditional_mat, ret_array=False):
    # make sure cond_dist is NOT transposed but original conditional distribution
    conditional_expectation = OrderedDict()
    conditional_variance = OrderedDict()
    for i1 in range(conditional_mat.shape[0]):
        c_e = 0.0
        for i2 in range(conditional_mat.shape[1]):
            c_e += i2*conditional_mat[i1, i2] # sum_i2 i2*p(i2 | i1)

        c_v = 0.0
        for i2 in range(conditional_mat.shape[1]):
            c_v += ((i2 - c_e)**2)*(conditional_mat[i1, i2])

        conditional_expectation[i1] = c_e
        conditional_variance[i1] = c_v

    if ret_array:
        conditional_expectation = np.array([v for k,v in conditional_expectation.items()])
        conditional_variance = np.array([v for k,v in conditional_variance.items()])
    return conditional_expectation, conditional_variance

def compute_conditional_std_unbinned(intensity_mat, stationary_dist, exclude_zero_weight=True, delta_time=1, ret_array=False):
    '''Computes Std(w_{t+1}|s_t), where w_{t+1} is unbinned'''
    intensity_mat = preproc_intensity(intensity_mat)
    states = stationary_dist.keys()

    conditional_std_unbinned = OrderedDict()
    for i1, s1 in enumerate(states):
        next_day_intensities = []
        for d1, d2 in zip(range(intensity_mat.shape[1]), range(intensity_mat.shape[1])[delta_time:]):
            prev_day_intensity = intensity_mat[:, d1]
            next_day_intensity = intensity_mat[:, d2]

            s1_idx = np.where((s1[0] <= prev_day_intensity) & (prev_day_intensity < s1[1]))[0]
            curr_next_day_intensities = next_day_intensity[s1_idx]
            if exclude_zero_weight:
                curr_next_day_intensities = curr_next_day_intensities[curr_next_day_intensities!=0]
            next_day_intensities.extend(list(curr_next_day_intensities))
        conditional_std_unbinned[s1] = np.std(next_day_intensities)
    if ret_array:
        conditional_std_unbinned = np.array([v for k,v in conditional_std_unbinned.items()])
    return conditional_std_unbinned

def weight_triggered_average(intensity_mat,
                            preproc_intensity_mat=True,
                            align_type='birth',
                            average_over_missing=False,
                            return_aligned_mat=True,
                            relevant_syn_idx=None):

    if preproc_intensity_mat:
        intensity_mat = preproc_intensity(intensity_mat)
    if relevant_syn_idx is None: # use all synapses
        relevant_syn_idx = range(intensity_mat.shape[0])

    aligned_mat = []
    for s in relevant_syn_idx:
        nonzero_idx = np.nonzero(intensity_mat[s, :])
        if align_type == 'birth':
            first_idx = np.amin(nonzero_idx)
            aligned_row = list(intensity_mat[s, first_idx:])
            num_append = (intensity_mat.shape[1] - len(aligned_row))
            aligned_row.extend([0]*num_append)
        elif align_type == 'death':
            last_idx = np.amax(nonzero_idx)
            aligned_row = list(intensity_mat[s, :last_idx+1])
            num_prepend = (intensity_mat.shape[1] - len(aligned_row))
            for _ in range(num_prepend):
                aligned_row.insert(0, 0)
        else:
            raise ValueError
        aligned_row = np.array(aligned_row)
        aligned_mat.append(aligned_row)
    aligned_mat = np.array(aligned_mat)

    # we will average over missing synapses and treat their weight as 0
    if average_over_missing:
        wta = np.mean(aligned_mat, axis=0)
        wta_sem = sem(aligned_mat, axis=0)
    else:
        wta = np.true_divide(aligned_mat.sum(0),(aligned_mat!=0).sum(0))
        aligned_mat_nan =  np.where(np.isclose(aligned_mat, 0), np.nan, aligned_mat)
        wta_sem = sem(aligned_mat_nan, nan_policy='omit')

    if return_aligned_mat:
        return wta, wta_sem, aligned_mat
    else:
        return wta, wta_sem

def compute_analytic_stationary(conditional_mat):
    w, vl = scipy.linalg.eig(conditional_mat, left=True, right=False)
    idx_max = np.argmax(w)
    analytic_stationary = vl[:, idx_max]
    analytic_stationary_dist = np.divide(analytic_stationary, np.sum(analytic_stationary))
    analytic_stationary_dist = analytic_stationary_dist.astype('float64')
    return analytic_stationary_dist

def markov_pipeline(intensity_mat,
                    bin_params=10,
                    plot_hist=False,
                    bins_touse=None,
                    bin_per_day=True,
                    exclude_00=False,
                    reverse=False,
                    compute_analytic_stationary_dist=True,
                    delta_time=1):

    if bins_touse is None:
        # only want bins of nonzero weight, as 0 will represent 0 synaptic weight
        all_bins = bin_data(intensity_mat,
                            bins_touse=bin_params,
                            plot_hist=plot_hist,
                            only_nonzero=True,
                            bin_per_day=bin_per_day)
        if bin_per_day:
            median_bins = np.median(all_bins, axis=0) # median across days
            # extending leftmost and rightmost bins to get all synaptic weights
            bins_touse = copy.deepcopy(median_bins)

            # extend bins to include all data (min and max)
            bins_touse[0] = np.amin(all_bins)
            bins_touse[-1] = np.amax(all_bins)
        else:
            bins_touse = copy.deepcopy(all_bins)

    stationary_dist, bin_counts_mat = compute_stationary_dist(intensity_mat, bins_touse=bins_touse)
    conditional_dist = compute_conditional_dist(intensity_mat, stationary_dist, exclude_00=exclude_00, reverse=reverse, delta_time=delta_time)

    analytic_stationary_dist = None
    if compute_analytic_stationary_dist:
        analytic_stationary_dist = compute_analytic_stationary(conditional_dist)

    aout = {}

    aout['bins_touse'] = bins_touse
    aout['bin_counts_mat'] = bin_counts_mat
    aout['stationary_dist'] = stationary_dist
    aout['conditional_dist'] = conditional_dist
    aout['analytic_stationary_dist'] = analytic_stationary_dist
    return aout

def bootstrap_markov_pipeline(intensity_mat,
                              bootstrap_size=None,
                              bootstrap_runs = 30,
                              replace=True,
                              run_exact = False,
                              save_keys = ['stationary_dist', 'conditional_dist', 'analytic_stationary_dist', 'bin_counts_mat', 'intensity_mat'],
                              seed_val=0, **markov_kwargs):
    if seed_val is not None:
        np.random.seed(seed_val)

    if bootstrap_size is None:
        bootstrap_size = intensity_mat.shape[0]

    results = {}
    for k in save_keys:
        results[k] = []
        if k == 'stationary_dist':
            results[k + '_dict'] = []

    br = 0
    num_internal_runs = bootstrap_runs
    while br < num_internal_runs:
        br += 1
        chosen_idx = np.random.choice(range(intensity_mat.shape[0]), size=bootstrap_size, replace=replace)
        chosen_intensity_mat = intensity_mat[chosen_idx]
        try:
            curr_aout = markov_pipeline(chosen_intensity_mat, **markov_kwargs)
        except:
            if run_exact:
                print('This bootstrap example resulted in 0 transition for some state, will retry.')
                num_internal_runs += 1 # increment runs to retry to have the exact number of successful runs as specified originally in bootstrap_runs
            else:
                print('This bootstrap example resulted in 0 transition for some state, will exclude this bootstrap example.')
            continue

        for k in save_keys:
            if k == 'stationary_dist':
                curr_aout[k + '_dict'] = curr_aout[k]
                results[k + '_dict'].append(curr_aout[k])
                curr_aout[k] = get_stationary_arr(curr_aout[k])
            if k != 'intensity_mat':
                results[k].append(np.expand_dims(curr_aout[k], axis=0))
            else:
                results[k].append(np.expand_dims(chosen_intensity_mat, axis=0))

    for k in save_keys:
        results[k] = np.concatenate(results[k], axis=0)
    return results

def compute_bootstrap_mean_std(m):
    # given a matrix of bootstrap runs x ..., computes mean and std, based off of pg. 5 of http://www.helsinki.fi/~rummukai/lectures/montecarlo_oulu/lectures/mc_notes5.pdf
    num_runs = m.shape[0]
    bootstrap_mean = np.mean(m, axis=0)
    bootstrap_std = np.sqrt((1.0/(num_runs - 1)) * np.sum(np.square(m - np.expand_dims(bootstrap_mean, axis=0)), axis=0))
    return bootstrap_mean, bootstrap_std

def joint_markov_pipeline(intensity_mat, bin_params=10, plot_hist=False, bins_touse=None, bin_per_day=True):
    if bins_touse is not None:
        # only want bins of nonzero weight, as 0 will represent 0 synaptic weight
        all_bins = bin_data(intensity_mat,
                            bins_touse=bin_params,
                            plot_hist=plot_hist,
                            only_nonzero=True,
                            bin_per_day=bin_per_day)
        if bin_per_day:
            median_bins = np.median(all_bins, axis=0) # median across days
            # extending leftmost and rightmost bins to get all synaptic weights
            bins_touse = copy.deepcopy(median_bins)

            # extend bins to include all data (min and max)
            bins_touse[0] = np.amin(all_bins)
            bins_touse[-1] = np.amax(all_bins)
        else:
            bins_touse = copy.deepcopy(all_bins)

    stationary_dist_joint, bin_counts_mat = compute_stationary_dist_joint(intensity_mat, bins_touse=bins_touse)
    conditional_dist_joint, conditional_bin_counts = compute_conditional_dist_joint(intensity_mat, stationary_dist_joint)

    analytic_stationary_dist_joint = compute_analytic_stationary(conditional_dist_joint)

    aout = {}

    aout['bins_touse'] = bins_touse
    aout['bin_counts_mat'] = bin_counts_mat
    aout['stationary_dist'] = stationary_dist_joint
    aout['conditional_dist'] = conditional_dist_joint
    aout['conditional_bin_counts'] = conditional_bin_counts
    aout['analytic_stationary_dist'] = analytic_stationary_dist_joint
    return aout

def get_continuous_val(state, nonzero_bins):
    if np.isclose(state, 0):
        return 0
    else:
        return np.random.uniform(nonzero_bins[state-1], nonzero_bins[state])

def get_continuous_arr(state_arr, nonzero_bins):
    cont_arr = []
    for s in state_arr:
        c = get_continuous_val(s, nonzero_bins)
        cont_arr.append(c)
    cont_arr = np.array(cont_arr)
    return cont_arr

def run_chain(dend=None,
              weights_agg=None,
              markov_kwargs={},
              num_synapses=None,
              num_steps=None,
              ret_continuous=True,
              start_distribution=None,
              exclude_zero_at_start=False,
              make_zero_absorbing=False,
              seed_val=0):
    if seed_val is not None:
        np.random.seed(seed_val)

    if (weights_agg is None) and (dend is not None):
        weights_agg = dend.weights_nonan

    aout = markov_pipeline(weights_agg, **markov_kwargs)
    if start_distribution is None:
        start_distribution = get_stationary_arr(aout['analytic_stationary_dist'])
    if exclude_zero_at_start:
        # sometimes, we want to sample from the start distribution conditioned on not being in a 0 state
        nonz_sum = np.sum([p for p_idx, p in enumerate(start_distribution) if p_idx > 0])
        start_distribution = np.divide(start_distribution, nonz_sum)
        start_distribution[0] = 0.0
    else:
        start_distribution = np.divide(start_distribution, np.sum(start_distribution))
    states = range(len(start_distribution))
    if num_synapses is None:
        num_synapses = weights_agg.shape[0]
    if num_steps is None:
        num_steps = weights_agg.shape[1] - 1

    transition_mat = aout['conditional_dist'] # p_{i, j} = prob(j | i)
    if make_zero_absorbing:
        # once we enter the zero state, we stay there
        transition_mat[0] = np.zeros(transition_mat[0].shape)
        transition_mat[0,0] = 1.0
    nonzero_weight_bins = aout['bins_touse']

    states_arr = np.random.choice(states, num_synapses, p=start_distribution, replace=True)
    state_matrix = [np.expand_dims(states_arr, axis=-1)]

    cont_arr = get_continuous_arr(states_arr, nonzero_weight_bins)
    cont_matrix = [np.expand_dims(cont_arr, axis=-1)]

    for _ in range(num_steps):
        next_state_probs = transition_mat[states_arr]
        states_arr = []
        cont_arr = []
        for s in range(num_synapses):
            curr_probs = next_state_probs[s]
            curr_state = np.random.choice(states, 1, p=curr_probs, replace=True)
            curr_state = np.squeeze(curr_state)
            states_arr.append(curr_state)

            curr_cont = get_continuous_val(curr_state, nonzero_weight_bins)
            cont_arr.append(curr_cont)

        states_arr = np.array(states_arr)
        cont_arr = np.array(cont_arr)

        state_matrix.append(np.expand_dims(states_arr, axis=-1))
        cont_matrix.append(np.expand_dims(cont_arr, axis=-1))

    state_matrix = np.concatenate(state_matrix, axis=-1)
    cont_matrix = np.concatenate(cont_matrix, axis=-1)

    if ret_continuous:
        return state_matrix, cont_matrix
    else:
        return state_matrix

def bin_equal_addmult(weights_mat_full,
                      compute_diff=True,
                      exclude_zero=False,
                      use_logs=False,
                      bin_size=14,
                      num_bins=None,
                      abs_val=True,
                      sq_within_bin=False,
                      exclude_last_day=True,
                      ret_bins=False):
    '''Rather than purely examining the conditional variance of the Markov chain, we bin the data in a more finegrained way
    to examine multiplicative dynamics, based on Figure 4c of http://www.jneurosci.org/content/jneuro/31/26/9481.full.pdf'''
    if exclude_last_day:
        weights_mat = weights_mat_full[:, :-1] # choose up to imaging day before last
    else:
        weights_mat = weights_mat_full
    flattened_weights = np.ravel(weights_mat)
    sorted_idx = np.argsort(flattened_weights)
    if exclude_zero:
        first_nonz = np.nonzero(np.sort(flattened_weights))[0][0]
        sorted_idx = sorted_idx[first_nonz:]
    #assert(len(sorted_idx) % bin_size == 0)
    if num_bins is None:
        assert(bin_size is not None)
        num_bins = len(sorted_idx) / bin_size
        a = 1 if (len(sorted_idx) % bin_size > 0) else 0
        print('Total {}, Num Bins {}'.format(len(sorted_idx), num_bins+a))
    else:
        assert(bin_size is None)
        bin_size = len(sorted_idx) / num_bins
        print('Total {}, Bin size {}'.format(len(sorted_idx), bin_size))

    bin_avg = []
    bin_sem = []
    bin_diff_avg = []
    bin_diff_var = []
    bin_diff_sem = []
    bins_touse = []
    for i in range(int(num_bins)+1):
        curr_bin_idx_arr = [sorted_idx[i*bin_size]]
        if (i == num_bins):
            # we append last two bins in bins_touse at the end
            if len(sorted_idx) % bin_size > 0:
                curr_bin_idx_arr.append(sorted_idx[-1])
                curr_bin_idx_rng = sorted_idx[i*bin_size:]

            else:
                curr_bin_idx_arr.append(sorted_idx[(i+1)*bin_size])
                curr_bin_idx_rng = sorted_idx[i*bin_size:(i+1)*bin_size]
        else:
            curr_bin_idx_rng = sorted_idx[i*bin_size:(i+1)*bin_size]

        within_bin = []
        within_bin_diff = []
        syn_idx, day_idx = np.unravel_index(curr_bin_idx_rng, weights_mat.shape)
        for s_i, s in enumerate(syn_idx):
            s_d = day_idx[s_i]
            s_w = weights_mat_full[s, s_d]
            if abs_val:
                s_w = np.abs(s_w)
            s_w_next = weights_mat_full[s, s_d + 1]
            if abs_val:
                s_w_next = np.abs(s_w_next)
            if use_logs:
                assert(exclude_zero is True)
                s_w = np.log10(s_w)
                s_w_next = np.log10(s_w_next)

            curr_within_bin = s_w
            if sq_within_bin:
                curr_within_bin = np.square(curr_within_bin)
            within_bin.append(curr_within_bin)
            if compute_diff:
                curr_within_bin_diff = s_w_next - s_w
            else:
                curr_within_bin_diff = s_w_next
            if abs_val:
                curr_within_bin_diff = np.abs(curr_within_bin_diff)
            within_bin_diff.append(curr_within_bin_diff)

        if len(within_bin) > 0: # only if there are weights in this bin
            within_bin_avg = np.mean(within_bin)
            within_bin_sem = sem(within_bin)
            within_bin_diff_avg = np.mean(within_bin_diff)
            within_bin_diff_var = np.var(within_bin_diff)
            within_bin_diff_sem = sem(within_bin_diff)

            bin_avg.append(within_bin_avg)
            bin_sem.append(within_bin_sem)
            bin_diff_avg.append(within_bin_diff_avg)
            bin_diff_var.append(within_bin_diff_var)
            bin_diff_sem.append(within_bin_diff_sem)

        if ret_bins:
            curr_bin_syn_idx, curr_bin_day_idx = np.unravel_index(curr_bin_idx_arr, weights_mat.shape)
            for curr_bin_s_i, curr_bin_s in enumerate(curr_bin_syn_idx):
                curr_bin_s_d = curr_bin_day_idx[curr_bin_s_i]
                curr_bin_s_w = weights_mat_full[curr_bin_s, curr_bin_s_d]
                bins_touse.append(curr_bin_s_w)

    bin_avg = np.array(bin_avg)
    bin_sem = np.array(bin_sem)
    bin_diff_avg = np.array(bin_diff_avg)
    bin_diff_var = np.array(bin_diff_var)
    bin_diff_sem = np.array(bin_diff_sem)
    if ret_bins:
        return bin_avg, bin_sem, bin_diff_avg, bin_diff_sem, bin_diff_var, bins_touse
    else:
        return bin_avg, bin_sem, bin_diff_avg, bin_diff_sem, bin_diff_var

def cross_correlation(delta, weight_matrix):
    assert(len(weight_matrix.shape) == 2) # synapse x day
    cc = 0.0
    total_timepoints = weight_matrix.shape[1]
    mid_point_idx = ((2*total_timepoints - 1) // 2) # index corresponding to delta of 0
    s_rng = np.arange(0, weight_matrix.shape[0])
    d_rng = np.arange(0, weight_matrix.shape[1] - np.abs(delta))
    for s in s_rng:
        cc += (np.correlate(weight_matrix[s], weight_matrix[s], mode='full')[int(mid_point_idx+delta)])
    cc = cc * (1.0/len(s_rng)) * (1.0/len(d_rng))
    return cc

def compute_velocity(weight_matrix):
    # V[t] = W[t] - W[t-1]
    V = []
    for i in range(1, weight_matrix.shape[1]):
        V.append(np.expand_dims(weight_matrix[:, i] - weight_matrix[:, i-1], axis=-1))
    V = np.concatenate(V, axis=-1)
    return V

def compute_deltaw(weight_matrix):
    # D = W - (1/T)\sum_tW[t], per synapse
    D = weight_matrix - np.mean(weight_matrix, axis=1, keepdims=True)
    return D
