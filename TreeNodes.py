# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import math
import mathutils
from mathutils import Vector
from mathutils.kdtree import KDTree

import numpy as np

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

    def growth_epoch(self, p_attr, D : float, d_i : float, d_k : float):
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
        n_nodes = len(self)
        # Create kd-tree of all nodes
        kdt = KDTree(n_nodes)
        [kdt.insert(n.location, i) for i, n in enumerate(self)]
        kdt.balance()
        kill = []   # List of all attraction points that need to be deleted
        new_nodes = []  # List of all newly generated nodes 
        # Build correspondence table between nodes and attr. points
        corr = defaultdict(list)
        for i, p in enumerate(p_attr):
            ind, dist = kdt.find(p)[1:]
            # Kill overgrown attr. points
            if dist <= d_k:
                kill.append(i)
            # Else assign close attr. points to nodes
            elif (dist <= d_i):
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
    
    def reduce_nodes(self, red_angle : float):
        # Get list of indices pointing to each nodes parent
        par_inds = [0] * len(self)
        for n, node in enumerate(self):
            for c_i in node.child_indices:
                par_inds[c_i] = n
        # Remove superfluous nodes and correct their parent's indices
        kill_list = []
        for n, node in enumerate(self[1:], 1):
            if len(node.child_indices) == 1:
                parent_ind = par_inds[n]
                child_ind = node.child_indices[0]
                parent = self[parent_ind]
                child = self[child_ind]
                vec_1 = (node.location - parent.location)
                vec_2 = (child.location - node.location)
                if vec_1.angle(vec_2) <= red_angle:
                    parent.child_indices.remove(n)
                    parent.child_indices.append(child_ind)
                    par_inds[child_ind] = parent_ind
                    kill_list.append(n)
        for k in reversed(kill_list):
            self[k] = None
        
        # Update indices
        corr = {}
        new_counter = 0
        for old_counter, node in enumerate(self):
            if node is not None:
                corr[old_counter] = new_counter
                new_counter += 1
        self[:] = [node for node in self if node is not None]
        for node in self:
            node.child_indices = [corr[ind] for ind in node.child_indices]