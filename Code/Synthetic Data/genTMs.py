''' 
    Code adapted from MaxEntTM by Paul Tune
    See: https://github.com/ptuls/MaxEntTM/tree/master

    The function HMC_exact(), which is used to generate truncated normal random variables, 
    is adapted from: https://github.com/aripakman/hmc-tmg/blob/master/HMC_exact.m 
    The python implementation is provided by: https://github.com/Kejione/pyDRTtools/blob/master/pyDRTtools/HMC.py 
    Some modifications were made to make program run with modulated gravity model 
'''

import numpy as np
from numpy import inf, pi
from numpy.random import randn
from numpy.linalg import cholesky
from numpy.matlib import repmat

def NCIM(): 

    return

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
    

    return TM, G

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
    
    if trough_mean == 1: 
        
        return x 
    
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

if __name__ == "__main__":
    TM, G = MGM(num_nodes=20, mean_traffic = 100, pm_ratio = 2, spatial_var = 100, temporal_var = 0.01,
                 interval = 15, num_days = 7, print = False, plot = False)