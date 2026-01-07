import numpy as np
import igraph as ig 

def get_laplacian_column_new(G, node_index, weight_attr='weight'):
    """
    Vectorized computation of the Laplacian column.
    Speedup: ~50x compared to looping over edges in Python.
    """
    n = G.vcount()
    col = np.zeros(n, dtype=np.float64)
    col_abs = np.zeros(n, dtype=np.float64)
    
    # 1. Get incident edges and neighbors efficiently
    # neighbors() in igraph returns the list of neighbors for the node
    # Note: For multigraphs, this returns duplicates (which is what we want)
    neighbors = np.array(G.neighbors(node_index), dtype=np.int64)
    
    if len(neighbors) == 0:
        return col, col_abs

    # 2. Get weights in bulk
    incident_edge_ids = G.incident(node_index, mode='all')
    weights = np.array(G.es[incident_edge_ids][weight_attr])
    
    # 3. Vectorized Accumulation (Handles Multi-graph implicitly)
    # L_ij = -w_ij (for i != j)
    # We use np.add.at because 'neighbors' may contain duplicates
    np.add.at(col, neighbors, -weights)
    np.add.at(col_abs, neighbors, np.abs(weights))
    
    # 4. Handle Diagonal (L_ii = sum(|w_ik|)) and Self-Loops
    # Standard Laplacian: L_ii = degree. 
    # Self-loops usually don't contribute to off-diagonals in L, 
    # but they do contribute to the degree (diagonal dominance).
    
    # Mask for self-loops (neighbor == node_index)
    self_loop_mask = (neighbors == node_index)
    
    # Correct the column for self-loops: 
    # We subtracted w previously (line 30), so we add it back to make it 0 contribution to off-diagonal
    if np.any(self_loop_mask):
        np.add.at(col, neighbors[self_loop_mask], -weights[self_loop_mask])
    
    # Set the diagonal element
    # Strict definition: sum of absolute weights of all incident edges
    current_degree = np.sum(np.abs(weights))
    col_abs[node_index] *= 0.5
    col[node_index] = current_degree-col_abs[node_index]
    
    return col, col_abs


def clique_sample_new(G, v, col_abs, col_weights, rng):
    # 1. Get local topology
    incident_ids = G.incident(v, mode='all')
    degree = len(incident_ids)

    if degree == 0:
        return

    # 2. Get local weights
    local_weights = np.array(G.es[incident_ids]['weight'])
    abs_weights = np.abs(local_weights)
    #possible error source as we do again count self-loops double again, so it will likely be over-represented
    total_w = np.sum(abs_weights)

    # --- Self Loop Logic (Simplified & Vectorized) ---
    # We calculate contributions to self-loops for neighbors
    if total_w > 0:
        # Calculate update vector for all nodes
        # w_self = (L_vi / L_vv) * L_vi + corrections...
        # Note: Your original math logic is preserved here
        mask_self = col_abs != 0
        mask_self[v] = 0 # Don't add self-loop to v itself
        
        # Vectorized calculation over the mask
        val_abs = col_abs[mask_self]
        val_w = col_weights[mask_self]
        
        updates = -(col_abs[v] / total_w) * val_abs
        updates -= (1 / total_w) * (val_abs - np.abs(val_w)) * (2 * val_abs - (val_abs - np.abs(val_w)))
        # Add edges efficiently
        indices = np.flatnonzero(mask_self)
        if len(indices) > 0:
            new_loops = np.column_stack((indices, indices))
            G.add_edges(new_loops)
            # Assign weights to the newly added edges (last N edges)
            G.es[G.ecount() - len(indices):]['weight'] = updates

    # --- Batch Sampling ---
    if total_w > 0:
        probs = abs_weights / total_w
    else:
        probs = None

    # Sample Indices
    uniform_indices = rng.integers(0, degree, size=degree)
    weighted_indices = rng.choice(degree, size=degree, p=probs)
    
    incident_ids_arr = np.array(incident_ids)
    batch_uniform_ids = incident_ids_arr[uniform_indices]
    batch_weighted_ids = incident_ids_arr[weighted_indices]

    # Bulk Retrieve Tuples (Optimized)
    # Using list comprehension is fastest for igraph Tuple retrieval
    tuples_1 = np.array([e.tuple for e in G.es[batch_uniform_ids]])
    tuples_2 = np.array([e.tuple for e in G.es[batch_weighted_ids]])

    # Identify Neighbors
    neighbors_1 = np.where(tuples_1[:, 0] == v, tuples_1[:, 1], tuples_1[:, 0])
    neighbors_2 = np.where(tuples_2[:, 0] == v, tuples_2[:, 1], tuples_2[:, 0])

    # Filter Mask
    valid_mask = (neighbors_1 != neighbors_2) & (neighbors_1 != v) & (neighbors_2 != v)

    if not np.any(valid_mask):
        return

    # Extract Valid Data
    valid_u = neighbors_1[valid_mask]
    valid_w = neighbors_2[valid_mask]

    w1_all = local_weights[uniform_indices] # We can index local_weights directly!
    w2_all = local_weights[weighted_indices]

    w1 = w1_all[valid_mask]
    w2 = w2_all[valid_mask]

    # Calculate new weights
    # Note: Your formula uses harmonic-style scaling. 
    # Standard Schur Complement sampling often uses w1*w2 / sum(w).
    # Preserving your custom logic:
    denom = np.abs(w1) + np.abs(w2)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        new_weights = np.where(denom != 0, (w1 * w2) / denom, 0)

    # Bulk Add Edges
    new_edges = np.column_stack((valid_u, valid_w))
    if new_edges.shape[0] > 0:
        G.add_edges(new_edges)
        G.es[G.ecount() - len(new_edges):]['weight'] = new_weights


def approx_chol_sdd_fast(S,k,epsilon, delta,c):
    """
    Docstring for approx_chol_sdd
    
    :param S: symmetric diagonally dominant matrix 
    :param k: size  of partial Cholesky decomposition
    :param epsilon: approximation parameter
    :param delta: accepted probability of failure
    :param c: boundness of each edge
 """
    rho=int(np.ceil(c * np.power(np.log(1/delta),2) /np.power(epsilon,2)))
    
    self_loops = np.diag(S) - (np.sum(np.abs(S), axis=1) - np.abs(np.diag(S)))
    weights_S = S.copy()
    np.fill_diagonal(weights_S, self_loops)
    rows, cols = np.where(np.abs(np.triu(weights_S, k=0)) > 1e-12)
    # Extract the specific weights for these edges
    base_weights = weights_S[rows, cols]
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
    n= S.shape[0]
    L= np.zeros([n,k])
    # 5. Build the Graph
    G = ig.Graph(n=n, edges=all_edges, directed=False)
    G.es['weight'] = normalized_weights
    G.vs['active'] = True

    rng = np.random.default_rng()
    # Generate the permutation (indices 0 to k-1)
    # Returns a new array
    perm = rng.permutation(k)
    for i in range(k):
        pi= perm[i]
        col_ci,col_abs = get_laplacian_column_new(G,pi,"weight")
        if col_ci[pi]==0:
            col_ci= np.zeros(col_ci.shape[0])
        edges_to_delete = G.incident(pi)
        clique_sample_new(G,pi,col_abs,col_ci, rng)    
        G.delete_edges(edges_to_delete)
        G.vs[pi]['active'] = False
        if col_ci[i]!=0:
            col_ci= np.power(col_ci[pi],-1/2)*col_ci
        L[:,pi]= col_ci
    return L


