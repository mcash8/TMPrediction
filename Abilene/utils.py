
'''
    Utility Functions 
'''
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
import networkx as nx
from matplotlib import pyplot as plt
import gurobipy as gb
import torch
from torch.autograd import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler
import os

''' 
    Build a fully connected network
'''

def fully_connected_network(num_nodes = 4, capacity = 100):

     # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(num_nodes))
    
    # Add edges with capacity
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                G.add_edge(i, j, capacity=capacity)

    return G


''' 
    Build a star network
'''

def star_network(num_nodes = 4, capacity = 100): 
    if num_nodes < 2:
        raise ValueError("The number of nodes must be at least 2 to create a star network.")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(num_nodes))
    
    # The central node is node 0
    central_node = 0
    
    # Add edges from the central node to all other nodes
    for node in range(1, num_nodes):
        G.add_edge(central_node, node, capacity=capacity)
        G.add_edge(node, central_node, capacity=capacity)

    return G

''' 
    Build a line network
'''

def line_network(num_nodes = 4, capacity = 100):
    
    # Create an undirected graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(range(num_nodes))
    
    # Add edges with specified capacity
    for i in range(num_nodes - 1):
        # Connect node i to node i+1
        G.add_edge(i, i + 1, capacity=capacity)
        G.add_edge(i+1, i, capacity=capacity)

    return G

''' 
    Cosine Similarity Function
'''

def cosine_similarity(A, B):    

    # Flatten the matrices
    vec1 = A.flatten()
    vec2 = B.flatten()
    
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return 1-similarity

''' 
    Build the Abilene Topology 
'''
def abilene_topo():
    G = nx.DiGraph()

    G.add_nodes_from(range(12))

    #add edges 
    with open('data/topo.txt', 'r') as f: 
        for line in f:
            parts = line.strip().split()

            # Extract node1, node2, and capacity
            node1 = int(parts[0])
            node2 = int(parts[1])
            capacity = float(parts[2])
            
            # Add the edges with the specified capacity
            G.add_edge(node1, node2, capacity=capacity)
            G.add_edge(node2, node1, capacity=capacity)
            
    return G

def abilene_topo_lossfn(): 
        ''' 
            If we use the original topology the MLU will be very small and differentiating is difficult.
            This topology scales the capacity values to be between 1 and 0.25.  
        '''
        G = nx.DiGraph()

        G.add_nodes_from(range(12))

        #add edges 
        with open('data/topo.txt', 'r') as f: 
            for line in f:
                parts = line.strip().split()

                # Extract node1, node2, and capacity
                node1 = int(parts[0])
                node2 = int(parts[1])

                if (int(parts[2]) == 9920000):  
                    capacity = 10
                elif (int(parts[2]) == 2480000): 
                    capacity = 2
                
                # Add the edges with the specified capacity
                G.add_edge(node1, node2, capacity=capacity)
                G.add_edge(node2, node1, capacity=capacity)
        
        return G


'''
    Functions for MCF
'''

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def  getCosts(G):
    '''
        Get cost for edges in G
    '''
    has_cost_attribute = all('cost' in G.edges[edge] for edge in G.edges())

    if has_cost_attribute:
        cost = nx.get_edge_attributes(G, 'cost')
    else:
        # If costs aren't specified, make uniform.
        nx.set_edge_attributes(G, name='cost', values=1)
        cost = nx.get_edge_attributes(G, 'cost')

    return cost

def getCapacity(G):
    '''
        Get capacity of edges
    '''
    return nx.get_edge_attributes(G, 'capacity')

def getLoadandFlowVars(m, V, arcs, cost):

    f = m.addVars(V, V, arcs, obj=cost, name='flow')
    l = m.addVars(arcs, lb=0.0, name='tot_traf_across_link')

    return f, l

def getLinkUtilization(m, l, f, arcs):

    '''Link util = sum over flows for each od pair '''
    
    # Link utilization = sum of flow 
    m.addConstrs((l[i, j] == f.sum('*', '*', i, j) for i, j in arcs), 'l_sum_traf',)

    return 

def getCapacityConstraint(m, l, capacity, arcs):
    '''Link utilzation can not exceed link capacity'''
    # Add capacity constraints 
    m.addConstrs(
        (l[i, j] <= capacity[i,j] for i, j in arcs),
        'traf_below_cap',
        )
    
    return

def getFlowConservationConstraint(m, D, V, f):
    ''' No flow gets left behind'''

    for s, t, u in cartesian_product(V, V, V):
        d = D[int(s-1), int(t-1)]

        if u==s:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==d, 'conserv')
        elif u==t:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==-d, 'conserv')
        else:
            m.addConstr(f.sum(s, t, u, '*')-f.sum(s, t, '*', u)==0, 'conserv')

    return 

def MinMaxLinkUtil(G,D, verbose = False): 

    # Create instance of optimizer model 
    m = gb.Model('netflow')
    V = np.array([i for i in G.nodes()])
    
    verboseprint = print
    if not verbose:
        verboseprint = lambda *a: None
        m.setParam('OutputFlag', False )
        m.setParam('LogToConsole', False )
    
    # Get link costs and capacity
    cost = getCosts(G)
    cap = getCapacity(G)

    arcs, capacity = gb.multidict(cap)

    # Create flow and load variables
    f, l = getLoadandFlowVars(m, V, arcs, cost)
    
    # Add link utilization constraints
    getLinkUtilization(m, l, f, arcs)
    
    # Add capacity constraints
    getCapacityConstraint(m, l, capacity, arcs)
    
    # Add flow conservation constraints 
    getFlowConservationConstraint(m, D, V, f)

    # Set objective to max-link utilization (congestion)
    max_cong = m.addVar(name='congestion')
    m.addConstrs(((cost[i,j]*l[i, j])/capacity[i,j]<= max_cong for i, j in arcs))
    m.setObjective(max_cong, gb.GRB.MINIMIZE)

    # Compute optimal solution
    m.optimize()

    # Print solution
    if m.status == gb.GRB.Status.OPTIMAL:
        f_sol = m.getAttr('x', f)
        l_sol = m.getAttr('x', l)
        m_cong = float(max_cong.x)


        if verbose == True:
            verboseprint('\nOptimal traffic flows.')
            verboseprint('f_{i -> j}(s, t) denotes amount of traffic from source'
                        ' s to destination t that goes through link (i, j) in E.')

            for s, t in cartesian_product(V, V):
                for i,j in arcs:
                    p = f_sol[s, t, i, j]
                    if p > 0:
                        verboseprint('f_{%s -> %s}(%s, %s): %g bytes.'
                                    % (i, j, s, t, p))

            verboseprint('\nTotal traffic through link.')
            verboseprint('l(i, j) denotes the total amount of traffic that passes'
                        ' through edge (i, j).'
            )

            for i, j in arcs:
                p = l_sol[i, j]
                if p > 0:
                    verboseprint('%s -> %s: %g bytes.' % (i, j, p))

            verboseprint('\nMaximum weighted link utilization (or congestion):',
                        format(m_cong, '.4f')
            )


        return m_cong
    else: 
        return


''' 
    Functions for data loading and processing
'''
def read_geant_data():
    df=pd.read_csv('C:\\Users\\marth\\OneDrive\\Desktop\\Baseline Prediction Methods & TM Exploring\\GEANT\\data\\geant-flat-tms.csv', header = None) #change path

    #convert series to date time data type 
    dates = df.iloc[:, 0]
    dates = pd.to_datetime(dates, format = '%Y-%m-%d-%H-%M')
    df[0] = dates

    #sort in ascending order
    df = df.sort_values(df.columns[0])
    df = df.set_index(df.columns[0])

    #get dataset 
    dataset = df.values.astype('float32') 
    return dataset

def read_abilene_data(read_week = False, week = 1): 
        
    ''' 
        Function to read the 24 weeks Abilene Dataset
    '''

    if read_week: 

        number = f"{week:02d}"
        
        # Path of file
        base_path = f"C:\\Users\\marth\\OneDrive\\Desktop\\Baseline Prediction Methods & TM Exploring\\Abilene\\data\\X{number}\\X{number}"  

        # Load file
        temp = np.loadtxt(base_path) 

        # Keep first 144 entries from each line 
        data = temp[:, :144]
        
        return data

    data = np.zeros((48384, 144))

    for i in range(1,25):

        # Format the number with leading zeros
        number = f"{i:02d}"
        
        # Path of file
        base_path = f"data/X{number}/X{number}"

        # Load file
        temp = np.loadtxt(base_path)

        # Keep first 144 entries from each line 
        temp = temp[:, :144]

        # Append data    
        data[(i-1)*2016:2016*i, :] = temp

    return data

def create_dataset(dataset, window_size): 
    dataX, dataY = [], [] 

    for i in range(len(dataset)-window_size): 
        a = dataset[i:i+window_size, :] 
        dataX.append(a) 
        dataY.append(dataset[i + window_size, :])

    return np.array(dataX), np.array(dataY) 

def train_test_split(dataset, train_ratio):

    train_size = int(len(dataset) * train_ratio)

    train_data = dataset[0:train_size,:] 
    test_data = dataset[train_size:len(dataset),:] 

    return train_data, test_data

def normalize_matrix(scaler, dataset):

    dataset_norm = np.zeros(dataset.shape)

    for i in range(len(dataset)):
        row = np.reshape(dataset[i, :], (dataset.shape[1],1))
        row = scaler.fit_transform(row)

        dataset_norm[i, :] = np.reshape(row, (dataset.shape[1],))

    return dataset_norm


def inverse_normalize_predictions(test_data, model_outputs):
    ''' 
        Perform inverse MinMax Normalization on the model predictions 
    '''

    inverse_preds = np.empty(model_outputs.shape)
    scaler = MinMaxScaler()

    for i in range(len(test_data)-10):    
        _ = scaler.fit_transform(test_data[i, :].reshape(-1, 1))

        inverse_preds[i, :] = scaler.inverse_transform(model_outputs[i,:].reshape(1, -1))

    return inverse_preds

def get_dataloader(inputs, labels, batch_size, num_workers, shuffle):

    ''' 
        Function to generate dataloaders for training and testing models 
    '''
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(inputs),
                                                 torch.Tensor(labels), 
                                                )

    dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=shuffle)

    return dataloader

''' 
    Base GRU Model 
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, _ = self.rnn(x, None)  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])  # return the last value
        out = self.sigmoid(out)
        
        return out

''' 
    ConvLSTMCell
'''
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

''' 
    Encoder Decoder LSTM
'''

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs

''' 
    Training and Testing Functions
'''

def train(model, train_loader, epochs, criterion, optimizer, device = torch.device("cpu")): 
    
    ''' 
        Training function for RNN to learn to predict TMs
    '''
    print('-----Begin Training------')
    track_losses = np.zeros(epochs)
    model.to(device)

    for epoch in range(epochs): 
        for inputs, targets in train_loader: 

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Pass data to LSTM
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Compute the gradient and update the network parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        training_loss = loss.item()
        track_losses[epoch] = training_loss 

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():10.4e}')

    return track_losses

def test(model, criterion, test_loader, testY): 
    
    ''' 
        Test RNN Model 
    '''
    
    model.eval() # Put model in eval mode 
    total_loss = 0.0 

    model_outputs = np.zeros((testY.shape[0], testY.shape[1]))
    test_loss = np.zeros((testY.shape[0], 1))

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader): 
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            model_outputs[idx, :] = outputs.numpy() # Store outputs
            
    average_loss = total_loss / len(test_loader)
    print(f'Average (Normalized) Test Loss: {average_loss:10.4e}')

    return model_outputs, test_loss

'''  
    Plotting Functions
'''

def plot_and_save_training_loss(training_loss, save_path='Figs\\training', fig_name = 'training_loss.png'):
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(save_path + '\\' + fig_name, dpi = 300)
    plt.close()  # Close the plot to free up memory

    return 


def plot_and_save_heatmap(label, prediction, num_nodes, save_path = 'Figs\\heat_maps', fig_name = 'heatmap.png'): 

    # Visual comparison of model outputs and ground truth
    sources = ['Src {}'.format(i) for i in range(0,num_nodes)]
    destinations = ['Dest {}'.format(i) for i in range(0,num_nodes)]

    sample_number = 20

    # Create a heatmap
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize = (12,5))
    im1 = ax1.imshow(np.reshape(label[sample_number+10, :], (num_nodes,num_nodes)), cmap = 'turbo') #scaled ground truth

    # Create colorbar
    cbar = ax1.figure.colorbar(im1, ax=ax1, shrink = 0.7)
    cbar.ax.set_ylabel('Demand (MBps)', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(destinations)), labels=destinations)
    ax1.set_yticks(np.arange(len(sources)), labels=sources)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=70, ha="right",
            rotation_mode="anchor")


    ax1.set_title("Ground Truth - %d Sample" % sample_number)

    im2 = ax2.imshow(np.reshape(prediction[sample_number, :], (num_nodes,num_nodes)), cmap = 'turbo') #scaled prediction

    # Create colorbar
    cbar = ax2.figure.colorbar(im2, ax=ax2, shrink = 0.7)
    cbar.ax.set_ylabel('Demand (MBps)', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax2.set_xticks(np.arange(len(destinations)), labels=destinations)
    ax2.set_yticks(np.arange(len(sources)), labels=sources)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=70, ha="right",
            rotation_mode="anchor")


    ax2.set_title("Predicted TM - %d Sample" % sample_number)

    # Compute normalized mean squared error 
    actual = label[sample_number+10, :]
    predicted = prediction[sample_number, :]

    mse = sum([(a - p) ** 2 for a, p in zip(actual, predicted)]) / len(actual)
    mean_actual = sum(actual) / len(actual)
    normalization_factor = sum([(a - mean_actual) ** 2 for a in actual]) / len(actual)
    nmse = mse / normalization_factor

    # Use axes-relative coordinates for annotation to ensure proper positioning
    fig.suptitle('NMSE: %0.3f' % nmse, va = 'center', fontsize = 'large', color = 'crimson')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}\\{fig_name}", dpi=300, bbox_inches='tight')
 
    plt.close()  # Close the plot to free up memory

    return 

def plot_and_save_ecdf(mlu_gt, mlu_preds, save_path, fig_name, title = 'CDF of MLU Bias'):

    bias = mlu_preds/mlu_gt[:-10]
    data_size = len(bias)

    data = np.reshape(bias, (len(mlu_preds),))

    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts*100)
    plt.plot(bin_edges[0:-1], cdf/np.max(cdf))
    plt.xlabel(r'Bias ($\frac{U^{\prime}}{U}$)')
    plt.ylabel(r'Cumulative Distribution of ($\frac{U^{\prime}}{U}$)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path + '\\' + fig_name, dpi = 300)
    plt.close()

    return 

def plot_and_save_mlu_compare(mlu_gt, mlu_preds, save_path, fig_name, title = 'Ground Truth vs. Predicted MLU'):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(mlu_gt[0:250], label = 'MLU Ground Truth')
    ax.plot(mlu_preds[0:250], label = 'MLU via Predicted TMs')

    ax.set_xlabel('Test Sample')
    ax.set_ylabel('MLU')
    ax.legend(loc = 'upper right')
    _ = ax.set_title(title)

    plt.savefig(save_path + '\\' + fig_name, dpi = 300)
    plt.close()

    return 

def plot_and_save_pdf(mlu_gt, mlu_preds, save_path = 'Figs\\pdfs', fig_name='PDF_mlu_abilene_global_mse.png',
                      title = 'PDF of MLU Bias' ):
    fig, ax1 = plt.subplots()
    bias = mlu_preds/mlu_gt[:-10]
    data = np.reshape(bias, (len(mlu_preds),))
    ax1.set_xlabel(r'Bias ($\frac{U^{\prime}}{U}$)')
    ax1.set_ylabel('Probability Density')
    ax1.hist(data, bins = 50, linewidth = 0.5, edgecolor = 'black', label = 'PDF MLU Bias', density = True)
    ax1.set_title(title)
    fig.savefig(save_path + '\\' + fig_name, dpi = 300)
    plt.close(fig)

    return 


def plot_and_save_diffmap(difference, num_nodes, save_path = 'Figs\\heat_maps', fig_name = 'heatmap.png'): 

    # Visual comparison of model outputs and ground truth
    sources = ['Src {}'.format(i) for i in range(0,num_nodes)]
    destinations = ['Dest {}'.format(i) for i in range(0,num_nodes)]

    sample_number = 20

    # Create a heatmap
    fig, ax1 = plt.subplots(nrows = 1, ncols=1, figsize = (12,5))
    im1 = ax1.imshow(np.reshape(difference[sample_number, :], (num_nodes,num_nodes)), cmap = 'turbo') #scaled ground truth

    # Create colorbar
    cbar = ax1.figure.colorbar(im1, ax=ax1, shrink = 0.7)
    cbar.ax.set_ylabel('Demand (MBps)', rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(destinations)), labels=destinations)
    ax1.set_yticks(np.arange(len(sources)), labels=sources)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=70, ha="right",
            rotation_mode="anchor")


    ax1.set_title("Predicted vs. Ground Truth Difference - %d Sample" % sample_number)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}\\{fig_name}", dpi=300, bbox_inches='tight')
    plt.close() 

    return 

'''
    Mlu baseline functions
'''

def mlu_on_preds(model_outputs, capacity=10, topo = 'abilene', num_nodes = 12):
    
    if topo == 'abilene': 
        G = abilene_topo()
        mlu_preds = np.empty((len(model_outputs), 1))

        for i in range(len(model_outputs)):

            D = model_outputs[i, :].reshape(12,12)
            np.fill_diagonal(D, 0)

            u = MinMaxLinkUtil(G, D*0.01)
            mlu_preds[i] = u

        Nans = np.any(np.isnan(mlu_preds)) # check for NaN values

    elif topo == 'fc':
        G = fully_connected_network(num_nodes, capacity)
        mlu_preds = np.empty((len(model_outputs), 1))

        for i in range(len(model_outputs)):

            D = model_outputs[i, :].reshape(12,12)
            np.fill_diagonal(D, 0)

            u = MinMaxLinkUtil(G, D)
            mlu_preds[i] = u

        Nans = np.any(np.isnan(mlu_preds)) # check for NaN valuesds
    
    return mlu_preds, Nans

''' 
    Denoising functions
'''

def moving_average_smoothing(data, window_size = 15):
    # Convert data to a pandas DataFrame for convenience
    time_series_df = pd.DataFrame(data)

    # Apply rolling mean to all columns
    smoothed_df = time_series_df.rolling(window=window_size, min_periods=1).mean()

    # Convert back to NumPy array if needed
    moving_avg = smoothed_df.values

    return moving_avg

def exp_moving_average_smoothing(data, alpha = 0.1): 
    
    # Convert data to a pandas DataFrame
    time_series_df = pd.DataFrame(data)

    # Apply EMA to all columns at once
    ema_df = time_series_df.ewm(alpha=alpha, adjust=False).mean()

    # Convert back to NumPy array if needed
    exp_ma = ema_df.values

    return exp_ma

def low_pass_filter(signal, cutoff_frequency = 50):

    # Perform Fourier Transform
    freq_data = np.fft.fft(signal)
    
    # Zero out high-frequency components
    freq_data[cutoff_frequency:] = 0
    freq_data[-cutoff_frequency:] = 0  # Mirror for negative frequencies
    
    # Inverse Fourier Transform to get back to time domain
    filtered_signal = np.fft.ifft(freq_data).real
    
    return filtered_signal


def butter_lowpass_filter(data, cutoff = 0.02, fs = 1, order=4):
    
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b, a, data)

def savitzky_golay_filt(data):

    # Apply Savitzky-Golay filter
    sg_filt_data = savgol_filter(data, window_length=50, polyorder=3, axis = 0)

    return sg_filt_data