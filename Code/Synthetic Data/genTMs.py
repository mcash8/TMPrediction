''' 
    Code adapted from MaxEntTM by Paul Tune
    See: https://github.com/ptuls/MaxEntTM/tree/master

    The function HMC_exact(), which is used to generate truncated normal random variables, 
    is adapted from: https://github.com/aripakman/hmc-tmg/blob/master/HMC_exact.m 
    The python implementation is provided by: https://github.com/Kejione/pyDRTtools/blob/master/pyDRTtools/HMC.py 
    Some modifications were made to make program run with modulated gravity model 


    To Do: 
        - Build a more general NCIM model that doesn't rely on row and column constraints 
          taken from abilene data
'''

import numpy as np
from numpy import inf, pi
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.linalg import cholesky
from numpy.matlib import repmat
import scipy.io

def NCIM(row, col, total, sigmasq): 
    ''' 
        Function to generate a series of traffic matrices via Non-conditionally independent model.
        We assume we know row and column constraints. 

        Input Variables: 
            - row: average total outgoing traffic from PoPs in each time interval
            - col: average total incoming traffic into PoPs in each time interval
            - total: total traffic in each time interval
            - sigmasq: noise variance for input into the truncated normal distribution

        Output: 
            - The script generates M traffic matrices, where M is the length of
            total, with the spatial size determined by the number of rows 
            row and number of columns of col. Note that row and col must have their
            lengths equal to that of total, i.e. they must be defined on M time
            intervals.
    '''
    

    # Sanity checks
    if total is None or len(total) == 0:
        raise ValueError("No total traffic provided")
    
    if (total.shape [0] > 1 & total.shape[1] > 1):
        raise ValueError("Total traffic should not be a matrix")
    
    
    if np.min(row) < 0:
        raise ValueError("Row entries must be non-negative")
    
    if np.min(col) < 0:
        raise ValueError("Column entries must be non-negative")
    
    Nrow, n_TMs = row.shape
    Ncol, m_TMs = col.shape
    
    if Nrow != Ncol:
        raise ValueError("Column length mismatch to row length")
    
    if n_TMs != m_TMs:
        raise ValueError("Column temporal length mismatch to row temporal length")
    
    if  total.shape[1] != n_TMs:
        raise ValueError("Total traffic length mismatch to row/column temporal length")
    
    if sigmasq < 0:
        raise ValueError("Noise variance must be non-negative")
    
    if np.min(total) < 0:
        raise ValueError("Total traffic must be non-negative")
    
    # Construct NCIM
    TM = np.zeros((Nrow, Ncol, n_TMs))
    for i in range(n_TMs):
        # Fanouts
        pU = row[:, i] / total[0,i]
        pV = col[:, i] / total[0,i]

        pU = np.expand_dims(pU, axis = 1)
        pV = np.expand_dims(pV, axis = 1)
        
        # Construct gravity-like model in each time slice
        U = HMC_exact(np.eye(Nrow), np.zeros((Nrow, 1)), np.eye(Nrow) * sigmasq, pU, True, 2, pU)
        V = HMC_exact(np.eye(Ncol), np.zeros((Ncol, 1)), np.eye(Ncol) * sigmasq, pV, True, 2, pV)
        
        TM[:, :, i] = total[0,i] * np.outer(U[:, 1], V[:, 1])
    
    return TM

def gen_NCIM(data = False): 

    mat = scipy.io.loadmat('tm_real.mat') # dictionary

    if data:
        tm_real = mat['data']
        tm_real = tm_real[:, :144]
        tm_real = np.transpose(tm_real, (1,0))
    else:
        tm_real = mat['tm_real']

    N = mat['N']
    N = int(N[0][0])
    n_TMs = mat['n_TMs']
    n_TMs = int(n_TMs[0][0])
    sigmasq = float(mat['sigmasq'][0][0]) # noise parameter \sigma^2 

    # turn into 3D array and compute the row and column marginals
    tm_tensor = np.reshape(tm_real, (N, N, n_TMs))
    r = np.sum(tm_tensor, axis=0) # Row marginals 
    c = np.sum(tm_tensor, axis=1) # Column marginals

    # Total traffic in each time interval
    T = np.sum(tm_real, axis = 0)
    T = T.reshape(1,-1)

    mean_row = np.mean(r, axis=1)  # Compute the mean across columns (axis=1)
    mean_col = np.mean(c, axis=1)  # Compute the mean across columns (axis=1)
    mean_T = np.mean(T)            # Compute the overall mean of T

    # Extract fourier coefficients and form sparse version, where Fourier
    # coefficients are selected via the nMDL criterion
    Tsparse, Ksparse = fourier_nmdl(T)

    # Need to construct the row and column spatiotemporal signals

    idNY = 8 # New York index (Python uses 0-based indexing adjust for )
    rs = np.zeros((N, n_TMs))  # Rows for all locations and time intervals
    cs = np.zeros((N, n_TMs))  # Columns for all locations and time intervals

    # Apply frequency sparsification for New York
    rs[idNY, :] = freq_sparsify(r[idNY, :], Ksparse)[0]  # Use [0] to get the sparse signal xs
    cs[idNY, :] = freq_sparsify(c[idNY, :], Ksparse)[0]

    # Convert hours to intervals (5 minutes per interval)
    m = 60 // 5

    # Time zone delays (everywhere except NY)
    D = m * np.array([0, 0, 1, 2, 1, 0, 1, 3, 3, 3, 0])  # Delays in measurement intervals

    # Locations except New York
    location = list(range(8)) + list(range(9, N))  # Exclude index 8 (New York)

    # Compute scaling factors
    rscale = mean_row / mean_row[idNY]
    cscale = mean_col / mean_col[idNY]

    # Apply scaling and time zone delay shifts
    for i, loc in enumerate(location):
        rs[loc, :] = np.roll(rscale[loc] * rs[idNY, :], D[i])
        cs[loc, :] = np.roll(cscale[loc] * cs[idNY, :], D[i])

    # Total traffic over all rows
    Ts = np.sum(rs, axis=0)
    Ts = Ts.reshape(1, -1)

    if data:
        TMncim = NCIM(r, c, T, sigmasq)
    else: 
        TMncim = NCIM(rs, cs, Ts, sigmasq)

    return TMncim, Tsparse, tm_real, mean_row, mean_col, sigmasq

def MGM(num_nodes=4, mean_traffic = 100, pm_ratio = 2, spatial_var = 100, temporal_var = 0.01,
                 interval = 15, num_days = 7, print = False, plot = False): 
    
    ''' 
        Function to generate a series of traffic matrices via modulated gravity model

        Input Variables: 
            - num_nodes: number of nodes (PoPs) in the networks
            - mean_traffic: mean total traffic value
            - pm_ratio: peak to mean ratio for generating sinusoidal signal
            - spatial_var: spaital variance
            - temporal_var: temporal variance 
            - interval: time interval between 'measured' TMs (min) ie: 5min, 15min, etc.
            - num_days: number of days to simulate TMs
            - print: boolean, print statistics of model
            - plot: boolean, plot simulated TMs 

        Output: 
    
    '''

    day = int((60/interval)*24) # number of TMs in a day
    t_ratio = 0.25*pm_ratio 
    diurnal_freq = 1/day # diuranl frequency inverse of number of TMs in a day
    n_TMs = num_days*day 

    total_traffic = peak_mean_cycle(diurnal_freq, n_TMs, mean_traffic, pm_ratio, t_ratio)
    

    if np.min(total_traffic) < 0: 
        total_traffic = total_traffic + abs(min(total_traffic))
        mean_traffic = np.mean(total_traffic)

    # Randomly generate incoming and outgoing total node traffic 
    fraction = np.random.rand(num_nodes, 1); 
    fraction /= np.sum(fraction)

    mean_row = fraction*mean_traffic # outgoing traffic

    fraction /= np.sum(fraction)
    mean_col = fraction*mean_traffic # incoming traffic 

    TM, G = modulated_gravity(mean_row, mean_col, total_traffic, spatial_var, temporal_var)

    if print: 
        print('to-do')

    if plot: 
        print('to-do')
    

    return TM, G, total_traffic

def peak_mean_cycle(freq, N, m, peak_mean, trough_mean):
    '''
        Generates a simple sinusoid with specified frequency, length, mean, and peak-to-mean ratio
        
        Input: 
            - freq: frequency of signal
            - N: length of signal
            - m: mean of signal
            - peak_mean: peak to mean ratio
            - trough_mean: trough to mean ratio

        Output: 
            - x: sinusoid signal 
    '''

    # sanity checks
    if peak_mean < 1: 
        raise Exception('Peak to mean ratio must be >= 1')
    
    if trough_mean > peak_mean:
        raise Exception('Trough to mean ratio must be less than or equal to peak to mean ratio')
    
    if m == 0: 
        # mean is zero
        x = peak_mean*np.sin(2*np.pi*freq*np.linspace(0, N-1, N))

        return x

    x = m*(peak_mean-1)*np.sin(2*np.pi*freq*np.linspace(0, N-1, N))+m
    
    if peak_mean == 1:
        return x.reshape(1,-1) 
    
    if trough_mean == 1: 
        
        return x.reshape(1,-1) 
    
    y = m * (trough_mean - 1) * np.sin(2 * np.pi * freq * np.linspace(0, N - 1, N)) + m
    x[x < m] = y[y < m]


    return x.reshape(1,-1) 

def modulated_gravity(mean_row, mean_col, total_traffic, spatial_var, temporal_var): 

    ''' 
        Generate a series of traffic matrices via the modulated gravity model. 

        Input:
            - mean_row: average total outgoing traffic from nodes
            - mean_col: average total incoming traffic from nodes
            - spatial_var: spatial variance component 
            - temporal_var: temporal variance component 

        Returns: 
            - M traffic matrices, where M is the length of the total_traffic.
            - G, the gravity model, for reference 
    '''

    # Sanity checks
    
    s,t = total_traffic.shape

    if s > 1: 
        total_traffic = total_traffic.conj().T
    
    if (s > 1 & t > 1):
        raise Exception('total traffic should not be a matrix')
    
    if min(mean_row) < 0:
        raise Exception('Row means should be non-negative')
    
    if min(mean_col) < 0:
        raise Exception('Column means should be non-negative')
    
    N = len(mean_row)

    # More sanity checks 
    if len(mean_col) != N: 
        raise Exception('Mismatch between row and col means')
    
    if spatial_var < 0: 
        raise Exception('Spatial variance should be non-negative')
    
    _, t  = mean_row.shape

    if t > 1: 
        mean_col = mean_col.conj().T
    
    if np.min(total_traffic) < 0: 
        raise Exception('Total traffic should be non-negative')

    n_TMs = total_traffic.shape[1]
    mean_total = np.mean(total_traffic)

    pU = mean_row/mean_total
    pV = mean_col/mean_total

    U = HMC_exact(np.eye(N), np.zeros((N,1)), np.eye(N)*spatial_var/mean_total**2, pU, True, 2, pU)
    V =  HMC_exact(np.eye(N), np.zeros((N,1)), np.eye(N)*spatial_var/mean_total**2, pV, True, 2, pV)

    normalized_mean = total_traffic/mean_total
    modulated = HMC_exact(np.eye(n_TMs), np.zeros((n_TMs,1)), temporal_var/mean_total**2, normalized_mean.conj().T,
                          True, 2, normalized_mean.conj().T)

    # Gravity Model 
    G = mean_total*np.matmul(np.reshape(U[:, 1], (-1,1)), np.reshape(V[:, 1], (-1,1)).conj().T)

    # Construct modulated gravity model
    temp1 = G.reshape(N**2, 1, order='F').copy()
    temp2  = np.reshape(modulated[:, 1].conj().T, (1,-1))
    temp = np.matmul(temp1, temp2)

    TM = temp.reshape(N,N,n_TMs, order='F').copy()

    return TM, G

def HMC_exact(F, g, M, mu_r, cov, L, initial_X):
        """
           This function returns samples from a truncated multivariate normal distribution.
           Reference: A. Pakman, L. Paninski, Exact hamiltonian monte carlo for truncated multivariate 
           Gaussians, J. Comput. Graph. Stat. 23 (2014) 518–542 (https://doi.org/10.48550/arXiv.1208.4118). 
           Inputs:
               F: m x d array (m is the number of constraints and d the dimension of the sample)
               g: m x 1 array 
               M: d x d array, which must be symmmetric and definite positive
               mu_r: d x 1 array 
               initial_X: d x 1 array, which must satisfy the constraint
               cov: condition to determine the covariance matrix, i.e., if cov == true, M is the covariance
                   matrix and mu_r the mean, while if cov == false M is the precision matrix and 
                   the log-density is -1/2*X'*M*X + r'*X
               L: number of samples desired
            Outputs:
               Xs: d x L array, each column is a sample being a sample from a d-dimensional Gaussian 
               with m constraints given by F*X+g >0 
        """
        
        # sanity check
        m = g.shape[0]
        if F.shape[0] != m:
            print("Error: constraint dimensions do not match")
            return

        # using covariance matrix
        if cov:
            mu = mu_r
            g = g + F@mu

            if M.shape == ():
                M = np.array([[M]])

            R = cholesky(M)
            R = R.T # change the lower matrix to upper matrix

            if R.shape == (1,1):
                F = F*R
            else: 
                F = F@R.T
            
            initial_X = initial_X -mu

            if R.shape == (1,1):
                initial_X = initial_X / R[0, 0]
            else: 
                initial_X = np.linalg.solve(R.T, initial_X)
            
        # using precision matrix
        else:
            r = mu_r
            R = cholesky(M)
            R = R.T # change the lower matrix to upper matrix
            mu = np.linalg.solve(R, np.linalg.solve(R.T, r))
            g = g + F@mu
            F = np.linalg.solve(R, F)
            initial_X = initial_X - mu
            initial_X = R@initial_X

        d = initial_X.shape[0]     # dimension of mean vector; each sample must be of this dimension
        bounce_count = 0
        nearzero = 1E-12
        
        # more for debugging purposes
        if (F@initial_X + g).any() < 0:
            print("Error: inconsistent initial condition")
            return

        # squared Euclidean norm of constraint matrix columns
        F2 = np.sum(np.square(F), axis=1)
        Ft = F.T
        
        last_X = initial_X
        Xs = np.zeros((d,L))
        Xs[:,0] = np.reshape(initial_X, -1)
        
        i=2
        
        # generate samples
        while i <=L:
            
            if i%1000 == 0:
                print('Current sample number',i,'/', L)
                
            stop = False
            j = -1
            # generate inital velocity from normal distribution
            V0 = np.random.normal(0, 1, (d,1))

            X = last_X
            T = pi/2
            tt = 0

            while True:
                a = np.real(V0)
                b = X

                fa = F@a
                fb = F@b

                U = np.sqrt(np.square(fa) + np.square(fb))
                # print(U.shape)

                # has to be arctan2 not arctan
                phi = np.arctan2(-fa, fb)

                # find the locations where the constraints were hit
                pn = np.array(np.abs(np.divide(g, U))<=1)
                
                if pn.any():
                    inds = np.where(pn)[0]
                    phn = phi[pn]
                    t1 = np.abs(-1.0*phn + np.arccos(np.divide(-1.0*g[pn], U[pn])))
                    
                    # if there was a previous reflection (j > -1)
                    # and there is a potential reflection at the sample plane
                    # make sure that a new reflection at j is not found because of numerical error
                    if j > -1:
                        if pn[j] == 1:
                            temp = np.cumsum(pn)
                            indj = temp[j]-1 # we changed this line
                            tt1 = t1[indj]
                            
                            if np.abs(tt1) < nearzero or np.abs(tt1 - pi) < nearzero:
                                # print(t1[indj])
                                t1[indj] = inf
                    
                    mt = np.min(t1)
                    m_ind = np.argmin(t1)
                    
                    # update j
                    j = inds[m_ind]
                     
                else:
                    mt = T
                
                # update travel time
                tt = tt + mt

                if tt >= T:
                    mt = mt- (tt - T)
                    stop = True

                # print(a)
                
                # update position and velocity
                X = a*np.sin(mt) + b*np.cos(mt)
                V = a*np.cos(mt) - b*np.sin(mt)

                if stop:
                    break
                
                # update new velocity
                qj = F[j,:]@V/F2[j]
                V0 = V - 2*qj*np.reshape(Ft[:,j], (-1,1))
                
                bounce_count += 1
        # Check if X satisfies the constraints before accepting it
            if (F@X +g).all() > 0:
                Xs[:,i-1] = np.reshape(X, (-1))
                last_X = X
                i = i+1
    
            else:
                print('hmc reject')    
        
            # need to transform back to unwhitened frame
        if cov:
            if R.shape == (1,1):
                Xs = R.T * Xs + repmat(mu.reshape(mu.shape[0],1),1,L)
            else: 
                Xs = R.T@Xs + repmat(mu.reshape(mu.shape[0],1),1,L)
        else:
            Xs =  np.linalg.solve(R, Xs) + repmat(mu.reshape(mu.shape[0],1),1,L)
        
        # convert back to array
        return Xs

def fourier_nmdl(S): 

    m, n = S.shape

    if m > 1 and n > 1:
        raise ValueError('Not a 1-dimensional signal')

    if m > 1: 
        S = S.T
    
    # If signal has odd length, add DC component
    if n % 2 != 0:
        S = np.append(S, np.mean(S))
    
    Slen = n

    # Preprocessing: center signal by removing mean
    Smean = np.mean(S)
    fS = np.fft.fft(S - Smean)
    indf = np.argsort(-np.abs(fS[0][:Slen // 2]), axis = 0)  # Descending order of Fourier coefficients

    lim = Slen // 4
    nMDL = np.zeros((1,lim))

    x = np.zeros((lim, Slen))  # Initialize sparse signal reconstructions

    # Search for least number of Fourier coefficients based on nMDL
    for k in range(1, lim + 1):

        fs = np.zeros(Slen // 2, dtype=complex)
        fs[indf[:k]] = fS[0,indf[:k]]
        fsparse = np.concatenate([fs, [fS[0, Slen // 2]], np.conj(fs[-1:0:-1])])

        x[k - 1, :] = np.real(np.fft.ifft(fsparse)) + Smean

        # Full Fourier, no time sparse component
        # Each Fourier component has 2 DOF: magnitude and phase
        residue = np.linalg.norm(S - x[k - 1, :])

        # Number of parameters
        p = 2 * k + 1

        # Compute nMDL criterion
        R = residue**2 / (Slen - p)
        nMDL[0, k - 1] = (0.5 * Slen * np.log(R)
                       + 0.5 * p * np.log((np.linalg.norm(S)**2 - residue**2) / (p * R))
                       + 0.5 * np.log(Slen - p) - 1.5 * np.log(p))

    # Choose k coefficients based on nMDL
    Knmdl = np.argmin(nMDL) + 1

    # Reconstruct sparse signal
    fs = np.zeros(Slen // 2, dtype=complex)
    fs[indf[:Knmdl]] = fS[0,indf[:Knmdl]]
    fsparse = np.concatenate([fs, [fS[0,Slen // 2]], np.conj(fs[-1:0:-1])])
    Snmdl = np.real(np.fft.ifft(fsparse)) + Smean

    return Snmdl, int(Knmdl)

def freq_sparsify(x, K):
    """
    Compute a frequency sparse signal with K coefficients from the original signal.

    Parameters:
    x (array-like): Input signal.
    K (int): Number of significant Fourier coefficients to keep.

    Returns:
    xs (numpy.ndarray): Frequency sparse signal.
    fsparse (numpy.ndarray): Sparse Discrete Fourier Transform of the signal.
    """
    # Length of the signal
    lx = len(x)
    
    # Compute the FFT of the mean-centered signal
    fx = np.fft.fft(x - np.mean(x))

    # Sort the magnitude of the first half of the FFT (excluding DC component) in descending order
    mag = np.abs(fx[:lx // 2])
    fxmag = np.sort(mag)[::-1]
    indf = np.argsort(mag)[::-1]  # Indices of sorted magnitudes in descending order

    # Keep only the top K Fourier coefficients
    fs = np.zeros(lx // 2, dtype=complex)
    fs[indf[:K]] = fx[indf[:K]]

    # Construct the sparse Fourier transform
    fsparse = np.zeros(lx, dtype=complex)
    fsparse[:lx // 2] = fs
    fsparse[lx // 2] = np.real(fx[lx // 2])  # Nyquist component
    fsparse[lx // 2 + 1:] = np.conj(fs[::-1][:-1])  # Conjugate symmetry, reverse and then exclude the last element
    fsparse[0] = np.sum(x)  # DC component

    # Compute the inverse FFT to get the sparse signal
    xs = np.real(np.fft.ifft(fsparse))

    return xs, fsparse


if __name__ == "__main__":

    # NCIM
    TMncim, Tsparse, tm_real, mean_row, mean_col, sigmasq  = gen_NCIM()

    # MGM - spatial and temporal var = sigmasq
    mean_Tsparse = np.mean(Tsparse)
    Tsparse = np.expand_dims(Tsparse, axis = 1)
    mean_row = np.expand_dims(mean_row, axis = 1)
    mean_col = np.expand_dims(mean_col, axis = 1)

    TMmod_gravity, _ = modulated_gravity(mean_row,mean_col,Tsparse,sigmasq, sigmasq)
    
    # Parameters for plotting
    dgreen = [0.05, 0.5, 0.05]  # RGB color for green
    dgrey = [0.65, 0.65, 0.65]  # RGB color for grey
    dred = [0.8, 0.1, 0.1] # RGB color for red
    fontsize = 18
    print_figures = False  # Set to save figures or not
    paper_position = [0, 0, 14, 12]  # Ignored for Python but can set figsize
    device = 'eps'  # File format for saving (optional)
    thick_line = 1.1
    line = 1
    outfile = 'maxent_pca_'  # Prefix for output file

    # Plot total sum
    plt.figure(figsize=(14, 12))  # Set figure size
    plt.plot(np.sum(tm_real, axis=0), color=dgrey, linewidth=thick_line, label='Real Traffic')  # Sum over axis 0
    plt.plot(np.sum(np.sum(TMmod_gravity, axis=0), axis=0), color=dred, label='Modulated Gravity Model')
    plt.plot(np.sum(np.sum(TMncim, axis=0), axis=0), color=dgreen, label='NCIM Model')
    
    n_TMs = TMncim.shape[2]
    # Customize plot
    plt.xlim([0, n_TMs])
    plt.xlabel('$t_k$', fontsize=fontsize)  # LaTeX-style label
    plt.ylabel('Gbps', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    # Save or show the plot
    if print_figures:
        plt.savefig(f'{outfile}.eps', format=device, bbox_inches='tight')
    else:
        plt.show()

    
    #TM, G, total_traffic = MGM(num_nodes = 20, mean_traffic = 100, pm_ratio = 2, spatial_var = 100, temporal_var = 0.1,
    #        interval= 15, num_days = 7)