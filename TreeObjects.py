# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
 
import numpy as np
import numpy.linalg as la

from numba import njit
from numba.typed import List as nbList

from dataclasses import dataclass
from typing import List

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
        
    def generate_mesh_ji_liu_wang(self):
        # Gather general data
        nodes = self.nodes
        tree_data = self.tree_data
        base_radius = tree_data.sk_base_radius
        min_radius = tree_data.sk_min_radius
        loop_dist = tree_data.sk_loop_distance
        
        joints = [nodes[0]]
        joints.extend([n for n in nodes if len(n.child_indices) > 1])
        
        limbs = []
        for joint in joints:
            for c_ind in joint.child_indices:
                new_limb = [joint]
                next_node = nodes[c_ind]
                while(len(next_node.child_indices) == 1):
                    new_limb.append(next_node)
                    next_node = nodes[next_node.child_indices[0]]
                new_limb.append(next_node)
                limbs.append(new_limb)
                
        nodes_per_limb = max([len(l) for l in limbs])
        n_limbs = len(limbs)
        l_locs = np.full((n_limbs, nodes_per_limb, 3), np.nan, dtype=np.float)
        l_weights = np.full((n_limbs, nodes_per_limb), np.nan, dtype=np.float)
        for l, limb in enumerate(limbs):
            for n, node in enumerate(limb):
                l_locs[l,n] = node.location
                l_weights[l,n] = node.weight_factor
        l_weights *= base_radius
        l_weights[l_weights < min_radius] = min_radius
        
        sweep(l_locs, l_weights, loop_dist)
        
@njit(cache=True)
def sweep(l_locs : np.array, l_weights : np.array, loop_dist : float):
    # res = nbList()
    for locs, weights in zip(l_locs, l_weights):
        pass