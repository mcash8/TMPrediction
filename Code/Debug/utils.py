'''
    Utility Functions 
'''

import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import gurobipy as gb
import random
import itertools as it


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
    Functions for data processing
'''

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

