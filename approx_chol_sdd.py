import numpy as np
import igraph as ig 


def get_laplacian_column(G, node_index, weight_attr='weight'):
    """
    Efficiently computes the column of the Laplacian corresponding to node_index.
    Works for multi-graphs by aggregating weights automatically.
    """
    # 1. Initialize zero vector (High speed, low memory)
    n = G.vcount()
    col = np.zeros(n, dtype=np.float64)
    col_abs =  np.zeros(n, dtype=np.float64)
    
    # 2. Get incident edges for the target node
    # This is O(deg(node)) - extremely fast compared to full matrix ops
    incident_edge_ids = G.incident(node_index, mode='all')
    
    # 3. Retrieve targets and weights in bulk
    edges = G.es[incident_edge_ids]
    
    # Handle unweighted graphs implicitly (weight=1)
    if weight_attr in G.es.attribute_names():
        weights = edges.get_attribute_values(weight_attr)
    else:
        weights = [1.0] * len(edges)
        
    # 4. Fill the column vector
    # We iterate only over the neighbors, not the whole graph
    current_degree = 0.0
    
    for edge, w in zip(edges, weights):
        # Identify the other node (neighbor)
        source, target = edge.tuple
        neighbor = target if source == node_index else source
        
        if neighbor == node_index:
            current_degree += np.abs(w)
            col_abs[neighbor] += np.abs(w)
            continue
            
        # Aggregate weight (Multi-graph support)
        # L_ij = - (sum of weights between i and j)
        col[neighbor] -= w
        col_abs[neighbor] += np.abs(w)
        current_degree += np.abs(w)
        
    # 5. Set the diagonal element (Degree)
    col[node_index] = current_degree
    
    return col, col_abs














def clique_sample(G,v,col_abs,col_weights,rng):
    # 1. Get local topology (Fast, O(deg(v)))
    incident_ids = G.incident(v, mode='all')
    degree = len(incident_ids)
    
   
    
    

    if degree == 0:
        return np.array([]), np.array([])

    # 2. Get local weights (Only for relevant edges)
    # This avoids reading the entire graph's weight list
    local_weights = np.array(G.es[incident_ids]['weight'])
    
    # 3. Prepare Probability Distribution (Vectorized)
    abs_weights = np.abs(local_weights)
    total_w = abs_weights.sum()
    
    mask_self=col_abs !=0
    mask_self[v] = 0

    weights_self_loop = np.zeros(col_abs.shape[0])
    weights_self_loop[mask_self] = col_abs[v]/total_w * col_abs[mask_self]
    weights_self_loop[mask_self] += 1/total_w *(col_abs[mask_self]-col_weights[mask_self])* (2* col_abs[mask_self]- (col_abs[mask_self]-col_weights[mask_self]))

    indices_self_loops = np.flatnonzero(mask_self)
    new_self_loops = np.column_stack((indices_self_loops, indices_self_loops)).astype(int)

    # 5. Bulk Add Edges
    # Only if there are edges to add (igraph throws error on empty list usually)
    if new_self_loops.shape[0] > 0:
        ecount_self = G.ecount()
        
        # Bulk add topology
        G.add_edges(new_self_loops)
    
        # Bulk assign weights to the new slice
        # This is much faster than iterating over the new edges
        G.es[ecount_self:]['weight'] = weights_self_loop[mask_self]
    

    if total_w > 0:
        probs = abs_weights / total_w
    else:
        # Fallback to uniform if all weights are zero
        probs = None 

    # 4. BATCH SAMPLING (The Performance Key)
    # instead of looping 'degree' times, we generate all samples at once.
    
    # Sample A: Uniform (deg(v) items)
    # We sample INDICES of the 'incident_ids' list, not the IDs directly yet
    # This is faster because we are sampling integers 0..k
    uniform_indices = rng.integers(0, degree, size=degree)
    
    # Sample B: Weighted (deg(v) items)
    weighted_indices = rng.choice(degree, size=degree, p=probs)
    
    # 5. Map back to real Edge IDs
    # converting list to array allows fast indexing
    incident_ids_arr = np.array(incident_ids)
    
    batch_uniform_ids = incident_ids_arr[uniform_indices]
    batch_weighted_ids = incident_ids_arr[weighted_indices]


    # 1. Bulk Retrieve Edge Tuples
    # igraph does not support vectorized tuple retrieval natively, so list comp is needed.
    # However, this is still much faster than processing logic inside the loop.
    tuples_1 = np.array([e.tuple for e in G.es[batch_uniform_ids]])
    tuples_2 = np.array([e.tuple for e in G.es[batch_weighted_ids]])

    # 2. Identify the "Other" Endpoint (Vectorized)
    # Logic: The sum of endpoints minus v must be the other node.
    # Example: Edge (v, u). Sum = v+u. (v+u) - v = u.
    neighbors_1 = tuples_1[:, 0] + tuples_1[:, 1] - v
    neighbors_2 = tuples_2[:, 0] + tuples_2[:, 1] - v

    # 3. Create Filter Mask (Vectorized)
    # Check all conditions at once:
    # A. Not a self-loop (neighbor != v)
    # B. Not the same node (neighbors_1 != neighbors_2)
    valid_mask = (
        (neighbors_1 != v) & 
        (neighbors_2 != v) & 
        (neighbors_1 != neighbors_2)
    )
    if not np.any(valid_mask):
        return 0
    


    # 4. Extract Valid Pairs
    valid_u = neighbors_1[valid_mask]
    valid_w = neighbors_2[valid_mask]

    w1_all = np.array(G.es[batch_uniform_ids]['weight'])
    w2_all = np.array(G.es[batch_weighted_ids]['weight'])
    
    # Apply the mask to weights so we match the valid new_edges
    w1_valid = w1_all[valid_mask]
    w2_valid = w2_all[valid_mask]
    
    # Calculate product (Vectorized multiplication)
    new_weights = w1_valid * w2_valid/(np.abs(w1_valid)+np.abs(w2_valid))
    
    # Stack them into an (M, 2) array for igraph
    # We cast to integer to ensure correct type for add_edges
    new_edges = np.column_stack((valid_u, valid_w)).astype(int)

    # 5. Bulk Add Edges
    # Only if there are edges to add (igraph throws error on empty list usually)
    if new_edges.shape[0] > 0:
        ecount_before = G.ecount()
        
        # Bulk add topology
        G.add_edges(new_edges)
    
        # Bulk assign weights to the new slice
        # This is much faster than iterating over the new edges
        G.es[ecount_before:]['weight'] = new_weights
    


def approx_chol_sdd(S,k,epsilon, delta,c):
    """
    Docstring for approx_chol_sdd
    
    :param S: symmetric diagonally dominant matrix 
    :param k: size  of partial Cholesky decomposition
    :param epsilon: approximation parameter
    :param delta: probability parameter
    :param c: boundness of each edge
    """
    rho= np.ceil(c * np.square(np.log(1/delta)) * np.power(epsilon,-2))
    rows, cols = np.where(S != 0)
    # Extract the specific weights for these edges
    base_weights = S[rows, cols]

    # 3. Create the "Split" Data
    # Repeat the edge list 'r' times
    # (r, 1) means repeat 'r' times vertically, 1 time horizontally
    base_edges = np.column_stack((rows, cols))
    all_edges = np.tile(base_edges, (rho, 1))

    # Repeat the weights 'r' times to match the edges
    all_weights = np.tile(base_weights, rho)

    # 4. Apply the 1/r Logic
    # This is a vectorized operation: divides all millions of weights at once
    normalized_weights = -all_weights / rho

    # 5. Build the Graph
    G = ig.Graph(n=S.shape[0], edges=all_edges, directed=False)
    G.es['weight'] = normalized_weights
    G.vs['active'] = True

    rng = np.random.default_rng()

    # Generate the permutation (indices 0 to k-1)
    # Returns a new array
    perm = rng.permutation(k)
    for i in range(k):
        pi= perm[k]
        col_ci,col_abs= get_laplacian_column(G,pi,"weight")
        if col_ci[pi]==0:
            col_ci= np.zeroes(col_ci.shape[0])
        G.delete_edges(G.incident(pi))
        G.vs[2]['active'] = False
        



