# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

# pylint: disable=relative-beyond-top-level, 

import bpy
import bmesh

import math
 
import numpy as np
import numpy.linalg as la

from numba import njit, prange, typeof
from numba import float64, int64
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
        bm = bmesh.new()    # pylint: disable=assignment-from-no-return
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
        n_joints = len(joints)
        
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
        max_limbs_per_joint = max([len(l) for l in start_nodes.values()]) + 1
        joint_map = np.full((n_joints, max_limbs_per_joint), -1, dtype=np.int64)
        for j, joint in enumerate(joints):
            j_id = id(joint)
            end = end_nodes.get(j_id, [-1])[0]
            joint_map[j,0] = end
            starts = start_nodes.get(j_id, [-1])
            joint_map[j,1:len(starts)+1] = np.asarray(starts)
        
        nodes_per_limb = max([len(l) for l in limbs])
        n_limbs = len(limbs)
        locs = np.full((n_limbs, nodes_per_limb, 3), np.nan, dtype=np.float)
        weights = np.full((n_limbs, nodes_per_limb), np.inf, dtype=np.float)
        for l, limb in enumerate(limbs):
            for n, node in enumerate(limb):
                locs[l,n] = node.location
                weights[l,n] = node.weight_factor
        weights *= base_radius
        weights[weights < min_radius] = min_radius
        weights[np.isinf(weights)] = -1.0
        
        verts = []
        squares = bmesh_ji_liu_wang(joint_map, locs, weights, loop_dist, interpolation_mode)
        for l in squares:
            for p in l:
                verts.append(p)
        # verts = np.array([e for sublist in verts for e in sublist]).reshape(-1,3)
        
        bm = bmesh.new()
        [bm.verts.new(v) for v in verts]
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        # # Take care of edges
        # bm.edges.index_update()
        # bm.edges.ensure_lookup_table()
        bm.to_mesh(self.bl_object.data)
        bm.free()

# Global constants
_global_coordinates = np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=np.float64)
_vertex_signs = np.transpose(np.array([[1,-1,-1,1], [1,1,-1,-1]]))
_1D_float_array = float64[:]
_2D_float_array = float64[:,:]
     
# TODO: Evaluate whether parallelization could get a speed increase for trees with many limbs
# TODO: Rewrite function to support AoT-compilation
@njit(fastmath=True, parallel=True)
def bmesh_ji_liu_wang(joint_map : np.array, node_locs : np.array, node_weights : np.array, loop_distance : float, interpolation_mode : str):
    # pylint: disable=not-an-iterable
    # General values
    n_limbs = len(node_locs)
    squares = nbList()
    
    ### Sweeping each limb separately
    for l in prange(n_limbs):
        limb_node_locs = node_locs[l]
        limb_node_weights = node_weights[l]
        n_nodes = np.count_nonzero(limb_node_weights > 0.0)
        # limb_locs = nbList() # Locations of nodes and points in between
        # limb_weights = nbList() # Weights of nodes and points in between
        
        ### Calculate locations and weights of points in between nodes
        # Interpolated weights
        sep_weight = nbDict.empty(key_type=int64, value_type=_1D_float_array)
        # Interpolated 3D locations
        sep_loc = nbDict.empty(key_type=int64, value_type=_2D_float_array)
        # Number of in-between-sections in this limb
        n_sections = n_nodes - 1
        for n in prange(n_sections):    # For each segment in the limb
            # Calculate vector and distance between the adjacent nodes
            prev_node_loc = limb_node_locs[n]
            next_node_loc = limb_node_locs[n+1]
            prev_node_weight = limb_node_weights[n]
            next_node_weight = limb_node_weights[n+1]
            diff = next_node_loc - prev_node_loc
            dist = la.norm(diff, 2)
            # Calculate number of evenly spaced separators
            n_separators = round(dist/loop_distance) - 1
            # Calculate positions of each separator on the line between 
            # the adjacent nodes for later interpolation
            offset = (dist - (n_separators - 1) * loop_distance) / 2
            # 1D positions of separators in this segment
            sec_sep_pos = np.empty((n_separators))
            # Weight of separators in this segment
            sec_sep_weight = np.empty((n_separators))
            # 3D location of separators in this segment
            sec_sep_loc = np.empty((n_separators, 3))
            for s in prange(n_separators):
                pos = (offset + s * loop_distance) / dist
                sec_sep_pos[s] = pos
                # Calculate weight by linear interpolation
                sec_sep_weight[s] = prev_node_weight * (1-pos) + next_node_weight * pos
                ### Calculate location
                # Either using linear interpolation
                if interpolation_mode == 'LIN':
                    sec_sep_loc[s] = prev_node_loc * (1-pos) + next_node_loc * pos
                # Or spline interpolation
                elif interpolation_mode == 'SPL':
                    pass
            sep_weight[n] = sec_sep_weight
            sep_loc[n] = sec_sep_loc
            squares.append(sec_sep_loc)
        
    return squares
        
    #     for n in prange(0, limb_locs, 2):
    #         half_n = int(n/2)
    #         locs[n] = limb_locs[half_n]
    #     for n in prange(1, limb_locs, 2):
    #         locs[n] = (locs[n-1] + locs[n+1]) / 2
    #     # Calculate tangent vectors for each node and bone inside the limb
    #     tangents = np.empty((n_tangents, 3), dtype=np.float64)
    #     for n in prange(0, n_tangents, 2):
    #         tangents[n] = locs[n+2] - locs[n]
    #     for n in prange(1, n_tangents, 2):
    #         tangents[n] = tangents[n-1] + tangents[n+1]
    #     tangent_lens = np.empty((n_tangents), dtype=np.float64)   # Length of each element of diff_vecs
    #     for n in prange(n_tangents):
    #         tangent_lens[n] = la.norm(tangents[n], 2)
    #         tangents[n] /= tangent_lens[n]
    #     # Calculate local coordinate systems
    #     # x is the direction of the respective tangent pointing town the limb
    #     local_coordinates = np.empty((n_tangents, 3, 3), dtype=np.float64)
    #     local_coordinates[:,0] = tangents
    #     for n in prange(n_tangents):
    #         # Special case: local x is parallel to global z
    #         # This float comparison is BAD! 
    #         # TODO: Replace with math.isclose as soon as it is supported in Numba
    #         lc = local_coordinates[n]
    #         if abs((lc[0] @ _global_coordinates[2]) - 1.0) < 0.0001:
    #             lc[1:] = _global_coordinates[:-1]
    #         else:
    #             # y is [global z (cross) x], z is [x (cross) y]
    #             lc[1] = np.cross(_global_coordinates[2], lc[0])
    #             lc[2] = np.cross(lc[0], lc[1])
    #             for vec in lc[1:]:
    #                 vec[:] /= la.norm(vec, 2)
    #     limb_squares = np.empty((n_tangents, 4, 3), dtype=np.float64)
    #     square_offets = locs[1:-1]
    #     for n in prange(n_tangents):
    #         lc = local_coordinates[n]
    #         s_offs = square_offets[n]
    #         for v in prange(4):
    #             sgn = _vertex_signs[v]
    #             limb_squares[n,v] = lc[2] * sgn[0] + lc[1] * sgn[1]
    #         limb_squares[n] += s_offs
    #     squares.append(limb_squares)
    # return(squares)
