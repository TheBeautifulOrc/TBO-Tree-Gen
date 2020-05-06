import bpy
import bmesh
import numpy as np
import math 

from .TreeNodes import Tree_Node, Tree_Node_Container

la = np.linalg
mask = np.ma

# Takes 1x3 numpy array (3D vector) and returns it's magnitude
def np_vec_magnitude(self, vec):
    return np.sqrt(np.sum(np.power(vec, 2)))

# Takes 1x3 numpy array (3D vector) and returns a normalized 1x3 numpy array
def np_vec_normalized(self, vec):
    mag = self.np_vec_magnitude(vec) 
    ret_val = vec/mag if mag != 0 else np.array([0,0,0])
    return ret_val

def generate_mesh(tree_nodes, base_radius):
    ### Data preparation
    n_nodes = len(tree_nodes)   # Number of total nodes         
    parents = np.zeros(n_nodes, dtype=np.int)   # List of parent indices 
    for p, parent in enumerate(tree_nodes):
        for c in parent.child_indices:
            parents[c] = p
    
    ### Copy relevant node data in numpy arrays
    node_locations = np.array([tn.location for tn in tree_nodes], dtype=np.float64) # (n_nodes*3) array of all nodes locations 
    node_radii = np.array([tn.weight_factor * base_radius for tn in tree_nodes], dtype=np.float64)  # n_nodes long array of all node radii
    
    ### Calculate arrays of matrix, that will make the information above faster to compute with
    adjacence_matrices = np.zeros((n_nodes, 7, 3))    # Up to six nodes can be adjacent to each node
    rel_adj_matrices = np.zeros((n_nodes, 7, 3))   # Same data as above, but all vectors are relative to the respective node
    # Calculate each entry of rel_adj_matrices
    for i in range(n_nodes):
        node_loc = node_locations[i]
        parent_loc = node_locations[parents[i]]
        adj_entry = np.zeros((7,3)) # Entry that will be written into adjacence_matrices
        rel_entry = np.zeros((7,3)) # Entry that will be written into rel_adj_matrices
        adj_entry[0] = parent_loc
        rel_entry[0] = parent_loc - node_loc
        for j, c in enumerate(tree_nodes[i].child_indices):
            child_loc = node_locations[c] 
            adj_entry[j+1] = child_loc
            rel_entry[j+1] = child_loc - node_loc
        adj_entry[6] = node_loc
        rel_entry[6] = node_loc
        adjacence_matrices[i] = adj_entry
        rel_adj_matrices[i] = rel_entry
    norm_matrices = rel_adj_matrices[:,0:6,:]/la.norm(rel_adj_matrices[:,0:6,:], 2, -1, keepdims=True)    # Normalized relative matrices
    
    ### Local coordinates
    cp = np.copy(norm_matrices) # Copy of nomalized vectors
    # Fill NaN-vecors with [1,0,0] (positive x)
    for i in range(3):
        cp_slice = cp[:,:,i]
        cp_slice[np.isnan(cp_slice)] = 1 if i == 0 else 0
    local_coordinates = np.zeros((n_nodes, 3, 3))
    # Local z = vector from parent to child 
    # Direction to parent is always down
    local_coordinates[:,2,:] = -cp[:,0,:]
    # Local x = local z 'cross' vector to first child (or global x if there are no children)
    local_coordinates[:,0,:] = np.cross(local_coordinates[:,2,:], cp[:,1,:])
    # Local y = local z 'cross' local x
    local_coordinates[:,1,:] = np.cross(local_coordinates[:,2,:], local_coordinates[:,0,:])
    local_coordinates[0] = [[1,0,0],[0,1,0],[0,0,1]]
    local_coordinates = np.divide(local_coordinates, la.norm(local_coordinates, 2, -1, keepdims=True))
    
    ### Assign directions to each neigbour 
    # Matrices that contain dot-products of norm_matrices and local_coordinates
    dot_matrices = np.zeros((n_nodes, 5, 3))    
    for i in range(n_nodes):
        dot_matrices[i] = norm_matrices[i][1:6,:] @ (local_coordinates[i])
    # Direction vectors
    # -1: unassigned, 0: x+, 1: x-, 2: y+, 3: y-, 4: z+, 5: z-
    directions = np.full((n_nodes, 6), -1, dtype=np.int8)   # Initialize unassigned
    directions[:,0].fill(5) # Parents are defined as z-
    
    for i in range(3):
        curr_col = dot_matrices[:,:,i]  # Column that is currently being evaluated 
        abs_curr_col = np.absolute(curr_col) # Absolute values of tis column
        abs_prev_col = np.absolute(dot_matrices[:,:,i-1])   # Absolute values of previous column
        abs_next_col = np.absolute(dot_matrices[:,:,i-2])   # Absolute values of next column (there are three columns)
        # Is the absolute value of this column greater than in the other two
        col_dominant = np.logical_and(np.greater_equal(abs_curr_col, abs_prev_col), np.greater(abs_curr_col, abs_next_col))
        col_positive = np.greater(curr_col, 0)
        directions[:,1:6] = mask.filled(mask.array(directions[:,1:6], mask=np.logical_and(col_dominant, col_positive), fill_value=i*2))
        directions[:,1:6] = mask.filled(mask.array(directions[:,1:6], mask=np.logical_and(col_dominant, ~col_positive), fill_value=i*2 + 1))

    # Create bmesh
    bm = bmesh.new()
    # Destroy bmesh
    bm.free()