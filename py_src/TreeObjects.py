# Copyright (C) 2019-2020 Luai "TheBeautifulOrc" Malek

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

# Global constants
_glob_orthonorm_basis = np.array([[0,0,1], [0,1,0], [1,0,0]])
_vertex_signs = np.transpose(np.array([[1,-1,-1,1], [1,1,-1,-1]]))
_1D_int_array = int64[:]
_1D_float_array = float64[:]
_2D_float_array = float64[:,:]

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
        locs = np.full((n_limbs, nodes_per_limb, 3), np.nan)
        weights = np.full((n_limbs, nodes_per_limb), np.inf)
        for l, limb in enumerate(limbs):
            for n, node in enumerate(limb):
                locs[l,n] = node.location
                weights[l,n] = node.weight_factor
        weights *= base_radius
        weights[weights < min_radius] = min_radius
        weights[np.isinf(weights)] = -1.0
        
        verts, connection_table = bmesh_ji_liu_wang(joint_map, locs, weights, loop_dist, interpolation_mode)
        # new_verts = []
        # for l in squares:
        #     for p in l:
        #         verts.append(p)
        # verts = new_verts
        
        # bm = bmesh.new()
        # [bm.verts.new(v) for v in verts]
        # bm.verts.index_update()
        # bm.verts.ensure_lookup_table()
        # # # Take care of edges
        # # bm.edges.index_update()
        # # bm.edges.ensure_lookup_table()
        # bm.to_mesh(self.bl_object.data)
        # bm.free()

# TODO: Rewrite function to support AoT-compilation
@njit(fastmath=True, parallel=True)
def bmesh_ji_liu_wang(joint_map : np.array, node_locs : np.array, node_weights : np.array, loop_distance : float, interpolation_mode : str):
    # pylint: disable=not-an-iterable
    # General purpose values
    n_limbs = len(node_locs)
    # Return values
    verts = nbList.empty_list(item_type=_1D_float_array)
    connection_table = nbList.empty_list(item_type=_1D_int_array)
    
    ### Sweeping each limb separately
    for l in prange(n_limbs):
        # Cut off all padding values
        n_nodes = np.count_nonzero(node_weights[l] >= 0.0)
        limb_node_locs = node_locs[l,:n_nodes]
        limb_node_weights = node_weights[l,:n_nodes]
        calc_in_between_locs(limb_node_locs, loop_distance, interpolation_mode)
        
    return verts, connection_table
    
@njit(fastmath=True, parallel=True)
def calc_in_between_locs(limb_node_locs, loop_distance, interpolation_mode):
    """Calculate locations of points in between nodes"""
    # pylint: disable=not-an-iterable
    locs = nbDict.empty(key_type=int64, value_type=_2D_float_array)
    # Number of in-between-sections in this limb
    n_nodes = len(limb_node_locs)
    n_sections = n_nodes - 1
    # Vectors and distances between nodes 
    diffs = limb_node_locs[1:] - limb_node_locs[:-1]
    dists = np.asarray([la.norm(diff, 2) for diff in diffs])
    # Number of points in between nodes necessary to keep distance 
    # between edge loops as close as possible to loop_distance 
    n_points = np.asarray([round(dist/loop_distance) - 1 for dist in dists])
    # Offsets between the nodes and neighboring interpolated points
    offsets = np.asarray([(dist - (npt - 1) * loop_distance) / 2 for dist, npt in zip(dists, n_points)])
    # Function variables that will be plugged into the interpolation functions
    positions = nbDict.empty(key_type=int64, value_type=_1D_float_array)
    for s in prange(n_sections):
        # Variables for this section
        npt = n_points[s]
        offset = offsets[s]
        dist = dists[s]
        pos = np.empty(npt)
        # Calculate each point separately
        for p in prange(npt):
            pos[p] = (offset + p * loop_distance) / dist
        positions[s] = pos
    # If the current limb consists of only one section (two nodes) 
    # interpolation will always be linear
    if n_sections < 2: 
        interpolation_mode = 'LIN'
    # Cubic spline interpolation
    if interpolation_mode == 'CUB':
        # Convenience vectors
        l_vec = dists[1:]   # Left of main diagonal
        r_vec = dists[:-1]  # Right of main diagonal
        d_vec = 2 * (l_vec + r_vec) # Main diagonal
        # Tridiagonal, diagonally dominant matrix (n_nodes x n_nodes)
        # In literature sometimes denoted as 'N' for hermetic cubic splines
        N = np.full((n_nodes, n_nodes), 0.0)
        # First and last entry are 1.0
        N[0,0] = 1.0
        N[-1,-1] = 1.0
        # Fill with convenience vectors
        for i in prange(n_sections-1):
            N[i+1,i] = l_vec[i]
            N[i+1,i+1] = d_vec[i]
            N[i+1,i+2] = r_vec[i]
        # Get inverse
        inv_N = la.inv(N)
        # Matrix on the righthand side of the spline-construction-equation
        # In literature sometimes denoted as 'R_sigma' for hermetic cubic splines
        R_sigma = np.empty((n_nodes, 3))
        # First and last entry are the slope of the first and last segment
        R_sigma[0] = diffs[0] / dists[0]
        R_sigma[-1] = diffs[-1] / dists[-1]
        # All other entries are calculated the following way
        for i in prange(n_sections-1):
            R_sigma[i+1] = 3 * ((diffs[i+1] * dists[i] / dists[i+1]) + (diffs[i] * dists[i+1] / dists[i]))
        # The matrix denoted as 'S' contains all the slope values necessary
        # to calculate the coefficients of the final spline functions
        S = inv_N @ R_sigma
        # Calculate the coefficients of the spline functions
        c1 = limb_node_locs[:-1]
        c2 = R_sigma[:-1]
        c3 = np.empty((n_sections, 3))
        c4 = np.empty((n_sections, 3))
        for i in prange(n_sections):
            c3[i] = (3 * limb_node_locs[i+1] 
                     - 3 * limb_node_locs[i] 
                     - 2 * S[i] * dists[i] 
                     - S[i+1] * dists[i]) / (dists[i]**2)
            c4[i] = (2 * limb_node_locs[i]
                     - 2 * limb_node_locs[i+1]
                     + S[i] * dists[i]
                     + S[i+1] * dists[i]) / (dists[i]**3)
        # TODO: Finish spline interpolation
    # Linear interpolation
    elif interpolation_mode == 'LIN':
        for s in prange(n_sections):
            npt = n_points[s]
            sec_positions = positions[s]
            sec_locs = np.empty((npt, 3))
            node_loc = limb_node_locs[s]
            diff = diffs[s]
            for p in prange(npt):
                sec_locs[p] = node_loc + sec_positions[p] * diff
            locs[s] = sec_locs
    
        
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
#         if abs((lc[0] @ _glob_orthonorm_basis[2]) - 1.0) < 0.0001:
#             lc[1:] = _glob_orthonorm_basis[:-1]
#         else:
#             # y is [global z (cross) x], z is [x (cross) y]
#             lc[1] = np.cross(_glob_orthonorm_basis[2], lc[0])
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