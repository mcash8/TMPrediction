import numpy as np 
from utils import * 
import sys

''' 
    Run mlu baseline on original, noisy data, not scaled
'''
def mlu_baseline(): 
    data = read_geant_data(read_week = False)
    start_idx = 0 
    end_idx = int((60/15)*24*21) # three weeks of data

    data = data[start_idx:end_idx]

    # For now, I don't need to compute the mlu for all 2016 data points
    # so I will only isolate the 404 test points to speed things up 
    training_len = int(len(data)*0.8)
    data = data[training_len:]
 
    num_nodes = 23
    G = fully_connected_network(num_nodes=num_nodes, capacity=1e7)

    mlu_gt = np.empty((len(data), 1))
    for i in range(len(data)):
        D = data[i, :].reshape(num_nodes, num_nodes)
        np.fill_diagonal(D,0)
        mlu_gt[i] = MinMaxLinkUtil(G,D)

    if np.any(np.isnan(mlu_gt)): 
        print('nans in mlus adjust parameters') 
        #sys.exit()
    
    np.save('mlu_baseline\\mlu_baseline_geant_fc.npy', mlu_gt)
    fig, ax = plt.subplots(figsize = (12,14))
    ax.plot(mlu_gt)
    plt.show()

    return 

''' 
    Run baselines on fully connected network. Data scaled by 1e5. 
    This is for the denoised approach since the 
'''
def mlu_baseline_scaled_original():
    print('running mlu baseline scaled original\n')
    data = read_geant_data()
    data = data/1e5
    start_idx = 0 
    end_idx = int((60/15)*24*21) # three weeks of data
    data = data[start_idx:end_idx]

    num_nodes = 23

    # Compute baseline mlu 
    G = fully_connected_network(num_nodes=23, capacity=10)
    mlu_gt = np.empty((len(data), 1))

    for i in range(len(data)):
        D = data[i, :].reshape(num_nodes, num_nodes)
        np.fill_diagonal(D,0)
        mlu_gt[i] = MinMaxLinkUtil(G,D)

        if i % 99 == 0: 
            print(i)

    if np.any(np.isnan(mlu_gt)): 
        print('nans in mlus adjust parameters') 
    

    np.save('mlu_baseline\\mlu_baseline_geant_scaled_fc.npy', mlu_gt)
    fig, ax = plt.subplots(figsize = (12,14))
    ax.plot(mlu_gt)
    plt.show()

    return

''' 
    Run baselines for denoised data. Data scaled by 1e5. 
'''

def mlu_baseline_scaled_denoised(method):

    methods_dict = {'moving_avg': moving_average_smoothing, 
                 'exp_moving_avg': exp_moving_average_smoothing, 
                 'low_pass_filt': low_pass_filter, 
                 'bw_filt': butter_lowpass_filter, 
                 'savgol': savitzky_golay_filt
                 }

    data = data/1e5
    num_nodes = 23

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
    G = fully_connected_network(num_nodes=23, capacity=10)
    mlu_gt = np.empty((len(data), 1))

    for i in range(len(data)):
        D = denoised_data[i, :].reshape(num_nodes, num_nodes)
        np.fill_diagonal(D,0)
        mlu_gt[i] = MinMaxLinkUtil(G,D)

    if np.any(np.isnan(mlu_gt)): 
        print('nans in mlus adjust parameters') 
        #sys.exit()

    save_name = f'mlu_baseline\\mlu_baseline_geant_fc_{method}.npy'
    np.save(save_name, mlu_gt)
    fig, ax = plt.subplots(figsize = (12,14))
    ax.plot(mlu_gt)
    plt.show()

    return 

if __name__ == "__main__":
    
    denoise = False
    methods = ['moving_avg', 'exp_moving_avg', 'low_pass_filt', 'bw_filt', 'savgol'] # denoising methods
    
    if denoise: 
        mlu_baseline_scaled_denoised(methods[4])
    else: 
        mlu_baseline_scaled_original()