# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import math
import mathutils

from mathutils import Vector
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

@dataclass
class Tree_Node:
    location : Vector
    parent_object : bpy.types.Object
    depth : int = 0
    weight : int = 1
    weight_factor : float = 0.0
    child_indices : List[int] = field(default_factory=list)
    
class Tree_Node_Container(list):
    # Returns a list of indices that point to the corresponding nodes parent.   
    def parent_indices(self):
        parent_indices = [0] * len(self)  # List of parent indices
        for i, node in enumerate(self):   # Setup parent indices
            for c in node.child_indices:
                parent_indices[c] = i
        return parent_indices
     
    # Reduces the vertex count of a list of tree nodes
    # Returns a new list of tree nodes
    def reduce_tree_nodes(self, tree_data):
        crit_angle = tree_data.vr_red_angle     # Minimum angle that needs to be preserved
        parent_indices = self.parent_indices()
        correspondence = {}     # Correspondences between indices in before and after reduction
        def mark_pending_kill(node, parent):
            # Replace parents connection to node with connection children
            parent.child_indices = [c for c in parent.child_indices if c != i]  # Remove node's entry
            parent.child_indices.extend([c for c in node.child_indices])    # Append child entries
            # Replace children's reference to node with reference to parent
            for c in node.child_indices:
                parent_indices[c] = parent_indices[i]
            self[i] = None    # Mark node as pending removal
        # Reduction algorithm
        for i, node in enumerate(self[1:]): # Foreach node except the root node (since it always remains unchanged) 
            i += 1  # Due to the nature of enumerate i is not matching the elements and must be adjusted
            parent = self[parent_indices[i]]
            clearance = (node.location - parent.location).length
            if clearance < (parent.weight_factor * tree_data.sk_base_radius) and len(node.child_indices) > 1:
                mark_pending_kill(node, parent)
            # Only nodes with exactly one child are candidates for angle absed reduction
            if len(node.child_indices) == 1:
                child = self[node.child_indices[0]]
                vec1 = Vector(node.location - parent.location)  # Vector between node's parent and itself
                vec2 = Vector(child.location - parent.location) # Vector between node's parent and it's child
                angle = vec1.angle(vec2, 0) # Calculate the angle between those two vectors
                if angle < crit_angle:  # If the angle is smaller that specified, the node is not essential
                    mark_pending_kill(node, parent)
        # Update correspondences 
        counter_new = 0
        for counter_old, node in enumerate(self):
            if node is not None:
                correspondence[counter_old] = counter_new
                counter_new += 1
        self[:] = [node for node in self if node is not None]   # Remove None objects from tree_nodes
        for node in self:
            node.child_indices = [correspondence[index] for index in node.child_indices]
            
    # Evaluates the nodes weight based on the number of recursive children it has
    def calculate_weights(self):
        n_nodes = len(self)   # Total number of tree nodes (relevant for weight factor calculation)
        for node in reversed(self):   # For each node (in reversed order)
            for c in node.child_indices:    # Look at each child
                node.weight += self[c].weight     # And add the child's (previoulsy calculated) weight to the nodes weight
            node.weight_factor = math.sqrt(node.weight / n_nodes)  # The weight factor is ratio between all nodes and the nodes that are connected to this particular node

    # Iterates the growth algorithm over the given tree nodes.
    # Returns 'True' if new nodes have been added.
    def iterate_growth(self, p_attr, tree_data, d_i_override=None):
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

    # Evaluates the depth (and thus the thickness) of every tree node
    def calculate_depths(self):
        for tn in self:   # For each tree node 
            if len(tn.child_indices) > 1:   # If the node is a branching point
                child_depth = tn.depth + 1  # The child branches shlould be of higher depth
            else:   # If there's only one or no child
                child_depth = tn.depth  # The child is a continuation of the current branch and should therefore have the same depth
            for child in tn.child_indices:
                self[child].depth = child_depth

    # Seperates a list of mixed nodes.
    # Returns a list of all nodes that have obj as parent.
    def separate_by_object(self, obj):
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