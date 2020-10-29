# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh

import math
 
import numpy as np
import numpy.linalg as la

from numba import njit, prange, float64, int64
from numba.typed import List as nbList, Dict as nbDict

from dataclasses import dataclass
from typing import List
from collections import defaultdict

from .TreeNodes import TreeNode, TreeNodeContainer
from .TreeProperties import TreeProperties
from .Utility import transform_points, quick_hull
        
class TreeObject:
    """
    Class used to combine all relevant data to create the trees mesh.
    
    In order to finalize a tree and create it's mesh in Blender this class 
    combines the data of a TreeNodecontainer and the corresponding Blender 
    object in addition to the TreeProperties variable. 
    """
    def __init__(self, 
                 bl_object : bpy.types.Object, 
                 nodes : TreeNodeContainer, 
                 tree_data : TreeProperties):
        """Constructor.
        
        Keyword arguments:
            bl_object : bpy.types.Object 
                Blender object that will become a tree
            nodes : TreeNodeContainer
                Nodes that make up this tree
            tree_data : TreeProperties 
                List of all properties of the tree that is being worked on
        """
        self.bl_object = bl_object
        self.nodes = nodes
        self.tree_data = tree_data
        
    def generate_skeltal_mesh(self):
        # Calculate vertices in local space 
        verts = [n.location for n in self.nodes]
        tf = self.bl_object.matrix_world.inverted()
        verts = transform_points(tf, verts)
        # Create bmesh
        bm = bmesh.new()
        # Insert vertices
        for v in verts:
            bm.verts.new(v)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        # Create edges
        for p, n in enumerate(self.nodes):
            for c in n.child_indices:
                bm.edges.new((bm.verts[p], bm.verts[c]))
        bm.edges.index_update()
        bm.edges.ensure_lookup_table()
        bm.to_mesh(self.bl_object.data)
        bm.free()
        
    # TODO: Implement spline interpolation as an alternative to linear interpolation (default)
    def generate_mesh(self):
        # Gather general data
        nodes = self.nodes
        tree_data = self.tree_data
        base_radius = tree_data.sk_base_radius
        min_radius = tree_data.sk_min_radius
        loop_dist = tree_data.sk_loop_distance
        interpolation_mode = tree_data.sk_interpolation_mode
        
        # Mark all nodes that are "joints"
        joints = [nodes[0]]
        joints.extend([n for n in nodes if len(n.child_indices) > 1])
        
        # Organize chains of nodes as "limbs"
        limbs = []
        for j, joint in enumerate(joints):
            for c_ind in joint.child_indices:
                new_limb = [joint]
                next_node = nodes[c_ind]
                while(len(next_node.child_indices) == 1):
                    new_limb.append(next_node)
                    next_node = nodes[next_node.child_indices[0]]
                new_limb.append(next_node)
                limbs.append(new_limb)
                
        # Create a map that shows which limbs are connected through which joints
        start_nodes, end_nodes = defaultdict(list), defaultdict(list)
        for i, l in enumerate(limbs):
            start_nodes[id(l[0])].append(i)
            end_nodes[id(l[-1])].append(i)
        n_joint_connections = max([len(l) for l in start_nodes.values()]) + 1
        
            
        nodes_per_limb = max([len(l) for l in limbs])
        n_limbs = len(limbs)
        locs = np.full((n_limbs, nodes_per_limb, 3), np.nan, dtype=np.float)
        weights = np.full((n_limbs, nodes_per_limb), np.nan, dtype=np.float)
        for l, limb in enumerate(limbs):
            for n, node in enumerate(limb):
                locs[l,n] = node.location
                weights[l,n] = node.weight_factor
        weights *= base_radius
        weights[weights < min_radius] = min_radius
        weights[np.isnan(weights)] = -1.0
        
        # verts = bmesh_ji_liu_wang(joint_map, locs, weights, loop_dist, interpolation_mode)
        # verts = np.array([e for sublist in verts for e in sublist]).reshape(-1,3)
        
        # bm = bmesh.new()
        # [bm.verts.new(v) for v in verts]
        # bm.verts.index_update()
        # bm.verts.ensure_lookup_table()
        # Take care of edges
        # bm.edges.index_update()
        # bm.edges.ensure_lookup_table()
        # bm.to_mesh(self.bl_object.data)
        # bm.free()

# Global constants
_global_coordinates = np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=np.float64)
_vertex_signs = np.transpose(np.array([[1,-1,-1,1], [1,1,-1,-1]]))
     
# TODO: Evaluate whether parallelization could get a speed increase for trees with many limbs
# TODO: Rewrite function to support AoT-compilation
@njit(fastmath=True, parallel=True)
def bmesh_ji_liu_wang(joint_map : nbDict, node_locs : np.array, node_weights : np.array, loop_distance : float, interpolation_mode : str):
    # General values
    n_limbs = len(l_locs)
    res_verts = nbList()
    # For each limb
    for l in prange(n_limbs):
        locs = l_locs[l]
        weights = l_weights[l]
        # Constants for each limb
        n_nodes = np.count_nonzero(weights > 0.0)
        n_locs = n_nodes * 2 - 1
        n_tangents = n_nodes * 2 - 3
        # Calculate locations in between nodes
        all_locs = np.empty((n_locs, 3), dtype=np.float64) # Locations of both nodes and centers of "bones"
        for n in prange(0, n_locs, 2):
            half_n = int(n/2)
            all_locs[n] = locs[half_n]
        for n in prange(1, n_locs, 2):
            all_locs[n] = (all_locs[n-1] + all_locs[n+1]) / 2
        # Calculate tangent vectors for each node and bone inside the limb
        tangents = np.empty((n_tangents, 3), dtype=np.float64)
        for n in prange(0, n_tangents, 2):
            tangents[n] = all_locs[n+2] - all_locs[n]
        for n in prange(1, n_tangents, 2):
            tangents[n] = tangents[n-1] + tangents[n+1]
        tangent_lens = np.empty((n_tangents), dtype=np.float64)   # Length of each element of diff_vecs
        for n in prange(n_tangents):
            tangent_lens[n] = la.norm(tangents[n], 2)
            tangents[n] /= tangent_lens[n]
        # Calculate local coordinate systems
        # x is the direction of the respective tangent pointing town the limb
        local_coordinates = np.empty((n_tangents, 3, 3), dtype=np.float64)
        local_coordinates[:,0] = tangents
        for n in prange(n_tangents):
            # Special case: local x is parallel to global z
            # This float comparison is BAD! 
            # TODO: Replace with math.isclose as soon as it is supported in Numba
            lc = local_coordinates[n]
            if abs((lc[0] @ global_coordinates[2]) - 1.0) < 0.0001:
                lc[1:] = global_coordinates[:-1]
            else:
                # y is [global z (cross) x], z is [x (cross) y]
                lc[1] = np.cross(global_coordinates[2], lc[0])
                lc[2] = np.cross(lc[0], lc[1])
                for vec in lc[1:]:
                    vec[:] /= la.norm(vec, 2)
        squares = np.empty((n_tangents, 4, 3), dtype=np.float64)
        square_offets = all_locs[1:-1]
        for n in prange(n_tangents):
            lc = local_coordinates[n]
            s_offs = square_offets[n]
            for v in prange(4):
                sgn = vertex_signs[v]
                squares[n,v] = lc[2] * sgn[0] + lc[1] * sgn[1]
            squares[n] += s_offs
        res_verts.append(squares)
    return(res_verts)