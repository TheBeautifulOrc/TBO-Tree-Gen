# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import math
import mathutils

import numpy as np

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

@dataclass
class TreeNode:
    """
    Data-collection used to store geometry information of an unfinished tree.
    
    During it's creation the tree is grown as a tree (the data-structure) of
    nodes. Each instance of this class will hold all relevant information 
    for one of these nodes. 
    
    Attributes: 
        location : np.array
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
    location : np.array
    parent_object : bpy.types.Object
    weight : int = 1
    weight_factor : float = 0.0
    child_indices : List[int] = field(default_factory=list)
    
class TreeNodeContainer(list):
    """
    Collection of many Tree_Nodes.
    
    During the growth of trees this container is used to keep track 
    of all the nodes that are generated during the process. 
    After separation by tree it provides utilities to calculate 
    node weights.
    """
    def parent_indices(self):
        """Returns a list of indices pointing to the parent of each node."""
        parent_indices = [0] * len(self)  # List of parent indices
        for i, node in enumerate(self):   # Setup parent indices
            for c in node.child_indices:
                parent_indices[c] = i
        return parent_indices
     
    def reduce_tree_nodes(self, tree_data):
        """
        Reduces the amount of TreeNodes.
        
        Reduces the the amount of tree nodes by deleting 
        nodes that change the overall geometry significantly.
        
        Keyword arguments:
            tree_data : TreeProperties 
                List of all properties of the tree that is being worked on
        
        Return value:
            A new, reduced list of tree nodes
        """
        crit_angle = tree_data.vr_red_angle     # Minimum angle that needs to be preserved
        parent_indices = self.parent_indices()
        # TODO: Clean up the unnecessary parts of this function
        #root_of_two = math.sqrt(2)
        
        def mark_pending_kill(node_index, node, parent):
            # Replace parents connection to node with connection children
            parent.child_indices = [c for c in parent.child_indices if c != node_index]  # Remove node's entry
            parent.child_indices.extend([c for c in node.child_indices])    # Append child entries
            # Replace children's reference to node with reference to parent
            for c in node.child_indices:
                parent_indices[c] = parent_indices[node_index]
            self[node_index] = None    # Mark node as pending removal
        
        # Reduction algorithm
        for i, node in enumerate(self[1:], 1): # Foreach node except the root node (since it always remains unchanged)
            parent_index = parent_indices[i]
            parent = self[parent_index]
            
            # clearance = (node.location - parent.location).length
            if False: # (len(node.child_indices) > 1 and
                #clearance < ((parent.weight_factor + node.weight_factor) * tree_data.sk_base_radius) * root_of_two):
                mark_pending_kill(i, node, parent)
            if False: #not tree_data.pr_enable_skinning:
                # Only nodes with exactly one child are candidates for angle based reduction
                if len(node.child_indices) == 1:
                    child = self[node.child_indices[0]]
                    vec1 = Vector(node.location - parent.location)  # Vector between node's parent and itself
                    vec2 = Vector(child.location - parent.location) # Vector between node's parent and it's child
                    angle = vec1.angle(vec2, 0) # Calculate the angle between those two vectors
                    if angle < crit_angle:  # If the angle is smaller that specified, the node is not essential
                        mark_pending_kill(i, node, parent)
        
        # Update correspondences
        correspondence = {}     # Correspondences between indices before and after reduction 
        counter_new = 0
        for counter_old, node in enumerate(self):
            if node is not None:
                correspondence[counter_old] = counter_new
                counter_new += 1
        self[:] = [node for node in self if node is not None]   # Remove None objects from tree_nodes
        for node in self:
            node.child_indices = [correspondence[index] for index in node.child_indices]
            node.child_indices = list(dict.fromkeys(node.child_indices))
            
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
        # Create kd-tree of all nodes
        n_nodes = len(self)
        kdt_nodes = mathutils.kdtree.KDTree(n_nodes)
        for i, node in enumerate(self):
            kdt_nodes.insert(node.location, i)
        kdt_nodes.balance()
        kill = []   # List of all attraction points that need to be deleted
        new_nodes = []  # List of all newly generated nodes 
        # Build correspondence table between nodes and attr. points
        corr = defaultdict(list)
        for p in p_attr:
            ind, dist = kdt_nodes.find(p)[1:]
            # Kill overgrown attr. points
            if dist <= d_k:
                kill.append(p)
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
            if n_vec.length < math.sqrt(2):
                n_vec = (corr[key][0] - loc).normalized()
            # Create child node
            new_nodes.append(Tree_Node((loc + D * n_vec.normalized()), parent_node.parent_object))
        self.extend(new_nodes)
        for p in kill:
            p_attr.remove(p)
        return (len(corr) > 0)

    def separate_by_object(self, obj):
        """
        Separates a TreeNodes by parent object.
        
        Keyword arguments:
            obj : bpy.types.Object
                Object whose children shall be returned
            
        Return value:
            Tree_Node_Container containing all nodes that have obj as parent
        """
        separate_nodes = Tree_Node_Container()
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