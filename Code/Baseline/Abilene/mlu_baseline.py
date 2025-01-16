import numpy as np 
from utils import * 
import sys

''' 
    Run baselines on fully connected network. Data scaled by 1e7
'''
def mlu_baseline_scaled_original():
    data = read_abilene_data(read_week = True, week = 1)
    data = data/1e7
    num_nodes = 12 

    # Compute baseline mlu 
    G = fully_connected_network(num_nodes=12, capacity=10)
    mlu_gt = np.empty((len(data), 1))

    for i in range(len(data)):
        D = data[i, :].reshape(num_nodes, num_nodes)
        np.fill_diagonal(D,0)
        mlu_gt[i] = MinMaxLinkUtil(G,D)

    if np.any(np.isnan(mlu_gt)): 
        print('nans in mlus adjust parameters') 
        #sys.exit()
    

    np.save('mlu_baseline\\mlu_baseline_abilene_fc.npy', mlu_gt)
    fig, ax = plt.subplots(figsize = (12,14))
    ax.plot(mlu_gt)
    plt.show()

    return

''' 
    Run baselines for denoised data. Data scaled by 1e7. 
'''


def mlu_baseline_scaled_denoised(method):

    methods_dict = {'moving_avg': moving_average_smoothing, 
                 'exp_moving_avg': exp_moving_average_smoothing, 
                 'low_pass_filt': low_pass_filter, 
                 'bw_filt': butter_lowpass_filter, 
                 'savgol': savitzky_golay_filt
                 }

    data = read_abilene_data(read_week = True, week = 1)
    data = data/1e7
    num_nodes = 12 

    if method == 'low_pass_filt':
        
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            denoised_data[:,i] = methods_dict[method](data[:, i])
    
    elif method == 'bw_filt':

        denoised_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            denoised_data[:,i] = methods_dict[method](data[:, i])

    else: 
        
        denoised_data = methods_dict[method](data)


    # Compute baseline mlu 
    G = fully_connected_network(num_nodes=12, capacity=10)
    mlu_gt = np.empty((len(data), 1))

    for i in range(len(data)):
        D = denoised_data[i, :].reshape(num_nodes, num_nodes)
        np.fill_diagonal(D,0)
        mlu_gt[i] = MinMaxLinkUtil(G,D)

    if np.any(np.isnan(mlu_gt)): 
        print('nans in mlus adjust parameters') 
        #sys.exit()

    save_name = f'mlu_baseline\\mlu_baseline_abilene_fc_{method}.npy'
    np.save(save_name, mlu_gt)
    fig, ax = plt.subplots(figsize = (12,14))
    ax.plot(mlu_gt)
    plt.show()

    return 

if __name__ == "__main__":
    
    denoise = True
    methods = ['moving_avg', 'exp_moving_avg', 'low_pass_filt', 'bw_filt', 'savgol'] # denoising methods
    
    if denoise: 
        mlu_baseline_scaled_denoised(methods[4])
    else: 
        mlu_baseline_scaled_original()