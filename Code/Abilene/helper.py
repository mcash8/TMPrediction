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
    Functions for plotting
'''

def generate_random_graph(num_nodes, probability, min_cap, max_cap):
    '''
        Function to generate a random graph according to a set of parameters. 
        The intended use is for testing the optimization function on different 
        configurations. 
    '''
    # Create an empty graph
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # All possible edges
    edges = list(it.permutations(range(0, num_nodes),2))
    #inidices = range(0, num_nodes)
    #random_subset = random.sample(indices, )
    #G.add_edges_from(edges)
    

    # Add edges randomly with a given probability
    for (i,j) in edges:
        if random.random() < probability:
            cost = 1
            capacity = random.randint(min_cap, max_cap)  # Random capacity between 1 and 10
            G.add_edge(i, j, capacity=capacity, cost = cost)
            G.add_edge(j, i, capacity=capacity, cost = cost)
       

    return G

def generate_demand_matrix(num_nodes, min_demand, max_demand):
    '''
        Function to generate a random demand matrix for input to the 
        optimization function
    '''

    D = np.random.random(size=(num_nodes, num_nodes))*(max_demand - min_demand) + min_demand
    np.fill_diagonal(D, 0)

    return np.round(D, 2)

def plot_graph(G, label_links = False):
    '''
        A function to visualize the graph that is the input to the optimization function. 
    '''
    pos = nx.spring_layout(G)  # Positions for all nodes
    capacities = nx.get_edge_attributes(G, 'capacity')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)

    if label_links: 
        nx.draw_networkx_edge_labels(G, pos, edge_labels=capacities)
   
    plt.title("Random Graph with Random Capacities")
    plt.show()

    return 

def TMGen(num_nodes, scale: float =  100, fixed_total: float = None) -> np.ndarray: 
    '''
        Generate a traffic matrix according to the gravity model. Create a NxN traffic matrix using the gravity model 
        with independent exponential distribution. 

        num_nodes: number of nodes in the network 
        scale: generates the exponential weight vecotrs. It's the inverse of the rate parameter. lambda = 1/scale
        fixed_total: the sum of all demands are scaled to the total demands in the network 
    '''

    traffic_in = np.array([np.random.default_rng().exponential(scale=scale, size = num_nodes)])
    traffic_out = np.array([np.random.default_rng().exponential(scale=scale, size = num_nodes)])
    
    traffic = (np.sum(traffic_in) + np.sum(traffic_out)) / 2 #sum(traffic_in) == sum(traffic_out) == traffic

    # probability matrix
    p_in = traffic_in / np.sum(traffic_in)
    p_out = traffic_out / np.sum(traffic_out)
    p_matrix = np.matmul(p_in.T, p_out)

    # traffic matrix
    traffic_matrix = p_matrix * traffic

    if fixed_total:
        multiplier = fixed_total / np.sum(traffic_matrix)
        traffic_matrix *= multiplier
    
    np.fill_diagonal(traffic_matrix, 0)

    return np.round(traffic_matrix,2)

import xml.etree.ElementTree as ET

def GeantTopo(): 
    '''
        Build networkx graph from geant topology xml file. The link capcity is assumed uniform according to the SNDLib file for Geant network. 
    '''
    # Create an empty graph
    G = nx.DiGraph()

    nodes = list(range(1,24))

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    tree = ET.parse('topology-anonymised.xml')
    root = tree.getroot()
    link_ids = []

    igp_links = root.find('.//igp/links')

    for link_element in igp_links:
        link_id = link_element.get('id')
        node1, node2 = link_id.split('_')
        
        G.add_edge(int(node1), int(node2), capacity = 40000.00, cost = 1) #assuming the cost is 1 for now
        G.add_edge(int(node2), int(node1), capacity = 40000.00, cost = 1) #assuming the cost is 1 for now

    return G

def map_capacity(deg_n1, deg_n2):
    '''
        Capacity mapping function -- maybe there's a more efficient way to do this? 
    '''
    if deg_n1 > 8 and deg_n2 > 8:
        return 10.00*1000  # 10 Gbps
    
    elif deg_n1 > 8 and deg_n2 == 8:
        return 10.00*1000  # 2.5 Gbps
    
    elif deg_n1 == 8 and deg_n2 == 8:
        return 2.5*1000 # 2.5 Gbps
    
    elif deg_n1 == 6 and deg_n2 == 6 or deg_n2 == 8 or deg_n2 == 10 or deg_n2 == 12: 
        return 2.5*1000
    
    elif deg_n1 == 4 and deg_n2 == 6 or deg_n2 == 8 or deg_n2 == 10 or deg_n2 == 122:
        return 2.5*1000  # 155 Mbps

    elif deg_n1 == 4 and deg_n2 == 4:
        return 155 # 155 Mbps
    
    elif deg_n2 > 8 and deg_n1 == 8:
        return 10.00*1000  # 2.5 Gbps
    
    elif deg_n2 == 8 and deg_n1 == 8:
        return 10.00*1000 # 2.5 Gbps
    
    elif deg_n2 == 6 and (deg_n1 == 6 or deg_n1 == 8 or deg_n1 == 10 or deg_n1 == 12): 
        return 2.5*1000
    
    elif deg_n2 == 4 and (deg_n1 == 6 or deg_n1 == 8 or deg_n1 == 10 or deg_n1 == 12):
        return 622
    
    elif deg_n2 == 4 and deg_n1 == 4:
        return 155 # 155 Mbps
    
    else:
        return 0  # Default capacity if none of the conditions are met

def GeantTopoNonUniform():
    '''
        Build networkx graph from geant topology xml file. The link capcity is not assumed uniform. 
        Instead, we assume that a node with more link connections has a higher link capacity. 
        See: Providing Public Intradomain Traffic Matrices to the Research Community
    '''
    # Create an empty graph
    G = nx.DiGraph()

    nodes = list(range(1,24))

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    tree = ET.parse('topology-anonymised.xml')
    root = tree.getroot()
    link_ids = []

    igp_links = root.find('.//igp/links')

    #add edges to nodes
    for link_element in igp_links:
        link_id = link_element.get('id')
        node1, node2 = link_id.split('_')
        
        G.add_edge(int(node1), int(node2), capacity = 0, cost = 1) #assuming the cost is 1 for now
        G.add_edge(int(node2), int(node1),  capacity = 0, cost = 1) #assuming the cost is 1 for now

    degrees_dict = dict(G.degree()) 

    
    '''
        I know ahead of time the node degree since I have the topology information. 
        The max is 12, the min is 4, theres also degrees of 6, 8, and 10. 

        see map_capacity for information
    '''

  
    for u, v in G.edges():
        deg_n1 = degrees_dict.get(u)
        deg_n2 = degrees_dict.get(v) 

        G[u][v]['capacity'] = map_capacity(deg_n1, deg_n2)

    return G

def plot_geant_graph(G, label_links = False):
    '''
        A function to visualize the graph that is the input to the optimization function. 
    '''
    pos = nx.spring_layout(G)  # Positions for all nodes
    capacities = nx.get_edge_attributes(G, 'capacity')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)

    if label_links: 
        nx.draw_networkx_edge_labels(G, pos, edge_labels=capacities)
   
    plt.title("GEANT Topology Anonymised")
    plt.show()

    return

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