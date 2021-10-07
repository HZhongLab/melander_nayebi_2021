import numpy as np

def get_hq_data(data):
    hq_mask = np.zeros(data.synapseMatrix.shape[0])
    syn, day = data.synapseMatrix.shape
    
    # Synapse Matrix
    for s in range(syn):
        for d in range(day):
            # print('high_quality' in list(data.fluorData[s,d].__dir__()))
            try:
                hq_mask[s] = data.fluorData[s, d].high_quality
            except:
                pass # Some older version files that were updated did not automatically have a 0 hq flag

    num_hq = np.sum(hq_mask)
    data.synapseMatrix = data.synapseMatrix[hq_mask == 1]
    data.synapseZ = data.synapseZ[hq_mask == 1]

    # Fluorescence matrix
    new_fluor_data = []

    for i, h in enumerate(hq_mask):
        if h == 1:
            new_fluor_data_row = []
            for d in range(day):
                new_fluor_data_row.append(data.fluorData[i, d])
            new_fluor_data.append(new_fluor_data_row)

    if len(new_fluor_data) == 0:
        print("No high quality synapses found for this dendrite.")
        raise ValueError


    data.fluorData = np.asarray(new_fluor_data)
    data.synapseID = data.synapseID[hq_mask == 1]

    return data


def split_unsure_synapses(syn_matrix, mtx, unsure_flag=3):
    # Disobey two day rule
    # syn_matrix = data.synapseMatrix

    def subroutine(m, original_row, row, r, nan_init=False):
        first = np.zeros(m.shape[1])
        second = np.zeros(m.shape[1])
        if nan_init:
            first = first + np.nan
            second = second + np.nan

        first[:r] = original_row[:r]
        second[r + 1 :] = original_row[r + 1 :]
        m = np.delete(m, row, axis=0)
        m = np.vstack([m, first])
        m = np.vstack([m, second])
        return m

    while len(np.where(syn_matrix == unsure_flag)[0]) > 0:
        rows, cols = np.where(syn_matrix == unsure_flag)
        row = rows[0]

        cur_row_synapse = syn_matrix[row, :]
        cur_row_mtx = mtx[row, :]

        r = np.where(cur_row_synapse == unsure_flag)[0][
            0
        ]  

        syn_matrix = subroutine(syn_matrix, original_row=cur_row_synapse, row=row, r=r)
        mtx = subroutine(mtx, original_row=cur_row_mtx, row=row, r=r)
    
    return mtx
