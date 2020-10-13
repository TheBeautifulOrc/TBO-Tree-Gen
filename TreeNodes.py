# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import math
import mathutils
from mathutils import Vector

import numpy as np
from scipy.spatial import cKDTree

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

sqrt_2 = math.sqrt(2)

@dataclass
class TreeNode:
    """
    Data-collection used to store geometry information of an unfinished tree.
    
    During it's creation the tree is grown as a tree (the data-structure) of
    nodes. Each instance of this class will hold all relevant information 
    for one of these nodes. 
    
    Attributes: 
        location : Vector
            Location of this node in 3D-space
        parent_object : bpy.types.Object 
            Blender object that will become the tree that this node is part of 
        weight : int
            Number of nodes that are children of this node 
        weight_factor : float 
            This nodes weight divided by the trees overall weight 
        child_indices : list(int)
            Indices of all nodes that are children of this node 
    """
    location : Vector
    parent_object : bpy.types.Object
    weight : int = 1
    weight_factor : float = 0.0
    child_indices : List[int] = field(default_factory=list)
    
class TreeNodeContainer(list):
    """
    Collection of many TreeNodes.
    
    During the growth of trees this container is used to keep track 
    of all the nodes that are generated during the process. 
    After separation by tree it provides utilities to calculate 
    node weights.
    """            
    def calculate_weights(self):
        """
        Caluclates node weights. 
        
        Evaluates the nodes weights based on the number of recursive children it has.
        """
        n_nodes = len(self)   # Total number of tree nodes (relevant for weight factor calculation)
        for node in reversed(self):   # For each node (in reversed order)
            for c in node.child_indices:    # Look at each child
                node.weight += self[c].weight     # And add the child's (previously calculated) weight to the nodes weight
            node.weight_factor = math.sqrt(node.weight / n_nodes)  # The weight factor is ratio between all nodes and the nodes that are connected to this particular node

    def iterate_growth(self, p_attr, tree_data, d_i_override=None):
        """
        One iteration step of the growth algorithm. 
        
        Iterates the growth algorithm over the given tree nodes by one step.
        
        Keyword arguments:
            p_attr : list(mathutils.Vector)
                List of all attraction points
            tree_data : TreeProperties 
                List of all properties of the tree that is being worked on
            d_i_override : float
                Manual override for influence distance of the algorithm 
                
        Returns value:
            'True' if new nodes have been added, otherwise 'False'
        """
        # Get tree_data values
        D = tree_data.sc_D
        d_i = d_i_override if d_i_override is not None else tree_data.sc_d_i * D
        d_k = tree_data.sc_d_k * D
        n_nodes = len(self)
        # Create kd-tree of all nodes
        kdt_nodes = mathutils.kdtree.KDTree(n_nodes)
        for i, node in enumerate(self):
            kdt_nodes.insert(node.location, i)
        kdt_nodes.balance()
        kill = []   # List of all attraction points that need to be deleted
        new_nodes = []  # List of all newly generated nodes 
        # Build correspondence table between nodes and attr. points
        corr = defaultdict(list)
        for i, p in enumerate(p_attr):
            ind, dist = kdt_nodes.find(p)[1:]
            # Kill overgrown attr. points
            if dist <= d_k:
                kill.append(i)
            # Else assign close attr. points to nodes
            elif (dist <= d_i or d_i == 0):
                corr[ind].append(p)
        # If the node has close attr. points
        # grow a child node
        for i, key in enumerate(corr):
            parent_node = self[key]
            parent_node.child_indices.append(n_nodes + i)
            # Calculate child location
            loc = parent_node.location
            n_vec = Vector((0.0,0.0,0.0))
            for p in corr[key]:
                n_vec += (p - loc).normalized()
            if n_vec.length < sqrt_2:
                n_vec = (corr[key][0] - loc).normalized()
            else:
                n_vec.normalize()
            # Create child node
            new_nodes.append(TreeNode((loc + D * n_vec), parent_node.parent_object))
        self.extend(new_nodes)
        for k in reversed(kill):
            p_attr.pop(k)
        return (len(corr) > 0)

    def separate_by_object(self, obj):
        """
        Separates a TreeNodes by parent object.
        
        Keyword arguments:
            obj : bpy.types.Object
                Object whose children shall be returned
            
        Return value:
            TreeNodeContainer containing all nodes that have obj as parent
        """
        separate_nodes = TreeNodeContainer()
        corr = {}
        corr_counter = 0
        for i, node in enumerate(self):
            if(node.parent_object is obj):
                separate_nodes.append(node)
                corr[i] = corr_counter
                corr_counter += 1
        for sn in separate_nodes:
            for i, c in enumerate(sn.child_indices):
                sn.child_indices[i] = corr[c]
        return separate_nodes