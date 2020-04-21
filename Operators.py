# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
import numpy as np
import mathutils
import math
import os 
import time
import multiprocessing as mp
import concurrent.futures as cf

from bpy.types import Operator
from mathutils import Vector
from collections import defaultdict
from .TreeNodes import Tree_Node

# Transfomrs a list of points according to the given 
# 4x4 transformation matrix using numpy.
# Returns the transformed vertices as a list. 
def transform_points(transf_matr, points):
    l  = len(points)
    np_points = np.array([element for tupl in points for element in tupl], dtype=np.float64)
    np_points.resize(l,3)
    np_tfm = np.array([element for tupl in transf_matr for element in tupl], dtype=np.float64)
    np_tfm.resize(4,4)
    np_retval = np.transpose((np_tfm @ np.pad(np_points.T, ((0,1), (0,0)), 'constant', constant_values=(1)))[0:3,:])
    retval = []
    for i in range(np_retval.shape[0]):
        retval.append(np_retval[i,:])
    return retval

# Generates points in/on an object using a particle system.
# Returns the points as a list 
def get_points_in_object(context, tree_data, temp_name="temp_part_sys"):
    obj = tree_data.shape_object
    seed = tree_data.seed
    n_points = tree_data.n_p_attr
    emission = tree_data.emit_p_attr
    distribution = tree_data.dist_p_attr
    even_distribution = tree_data.even_dist_p_attr
    use_modifier_stack = tree_data.use_shape_modifiers
    obj.modifiers.new(name=temp_name, type='PARTICLE_SYSTEM')
    ps = obj.particle_systems[-1]
    ps.seed = seed
    ps.settings.type = 'EMITTER'
    ps.settings.emit_from = emission
    ps.settings.count = n_points
    ps.settings.use_modifier_stack = use_modifier_stack
    ps.settings.distribution = distribution
    ps.settings.use_even_distribution = even_distribution
    ps = obj.evaluated_get(context.evaluated_depsgraph_get()).particle_systems[-1]
    np_arr = np.array([element for particle in ps.particles for element in particle.location], dtype=np.float64)
    ps = obj.modifiers.get(temp_name)
    obj.modifiers.remove(ps)
    np_arr.resize(n_points,3)
    arr = []
    for i in range(np_arr.shape[0]):
        arr.append(Vector((np_arr[i,:])))
    return arr

# Iterates the growth algorithm over the given tree nodes.
# Returns 'True' if new nodes have been added.
def iterate_growth(growth_nodes, p_attr, tree_data, d_i_override=None):
    # Get tree_data values
    D = tree_data.sc_D
    d_i = d_i_override if d_i_override else tree_data.sc_d_i * D
    d_k = tree_data.sc_d_k * D
    # Create kd-tree of all nodes
    n_nodes = len(growth_nodes)
    kdt_nodes = mathutils.kdtree.KDTree(n_nodes)
    for i, node in enumerate(growth_nodes):
        kdt_nodes.insert(node.location, i)
    kdt_nodes.balance()
    kill = []   # List of all attraction points that need to be deleted
    new_nodes = []  # List of all newly generated nodes 
    # Build correspondence table between nodes and attr. points
    corr = defaultdict(list)
    for p in p_attr:
        vec, ind, dist = kdt_nodes.find(p)
        # Kill overgrown attr. points
        if dist <= d_k:
            kill.append(p)
        # Else assign close attr. points to nodes
        elif (dist <= d_i or d_i == 0):
            corr[ind].append(p)
    # If the node has close attr. points
    # grow a child node
    for i, key in enumerate(corr):
        parent_node = growth_nodes[key]
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
    growth_nodes.extend(new_nodes)
    for p in kill:
        p_attr.remove(p)
    return (len(corr) > 0)

# Evaluates the depth (and thus the thickness) of every tree node
def calculate_depths(tree_nodes):
    for tn in tree_nodes:   # For each tree node 
        if len(tn.child_indices) > 1:   # If the node is a branching point
            child_depth = tn.depth + 1  # The child branches shlould be of higher depth
        else:   # If there's only one or no child
            child_depth = tn.depth  # The child is a continuation of the current branch and should therefore have the same depth
        for child in tn.child_indices:
            tree_nodes[child].depth = child_depth

# Seperates a list of mixed nodes.
# Returns a list of all nodes that have obj as parent.
def separate_nodes(mixed_nodes, obj):
    separate_nodes = []
    corr = {}
    for i, mn in enumerate(mixed_nodes):
        if(mn.parent_object is obj):
            separate_nodes.append(mn)
            corr[i] = len(separate_nodes) - 1
    for sn in separate_nodes:
        for i, c in enumerate(sn.child_indices):
            sn.child_indices[i] = corr[c]
    return separate_nodes

# Adds and applies a skin modifier to the skeletal mesh
def skin_skeleton(context, obj, tree_nodes, tree_data, temp_name="temp_skin_mod"):
    # Initialize skin modifier
    sk_mod = obj.modifiers.new(name=temp_name, type='SKIN')     # Create the modifier
    sk_mod.use_x_symmetry = False   #
    sk_mod.use_y_symmetry = False   # Disable symmetry options
    sk_mod.use_z_symmetry = False   #
    sk_mod.branch_smoothing = tree_data.sk_smoothing    # Set the branch smoothing to the desired value
    # Adjust MeshSkinVertices
    obj.data.skin_vertices[0].data[0].use_root = True
    for i, v in enumerate(obj.data.skin_vertices[0].data):
        weight_factor = tree_nodes[i].weight_factor
        rad = tree_data.sk_base_radius * weight_factor
        if rad < tree_data.sk_min_radius:
            rad = tree_data.sk_min_radius
        v.radius = (rad, rad) 
    context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=temp_name)

def _evaluate_branch(child_index, tree_nodes, parent, crit_angle):
    child = tree_nodes[child_index]   # Get the child node
    something_to_preserve = False
    while(not something_to_preserve):
        if len(child.child_indices) != 1:   # If the child, we are looking at, is a branch or endpoint, it must be preserved and the routine can end
            something_to_preserve = True                         
        else:   # Else we have to further evaluate if the child should be preserved
            grandchild = tree_nodes[child.child_indices[0]]     # Take a look at the child's child
            # Calculate the angle between the line segments [parent;child] and [parent;grandchild]
            angle = (parent.location - child.location).angle((parent.location - grandchild.location), 0)
            if angle >= crit_angle:     # If the angle is larger than specified by the user, the child must be preserved
                something_to_preserve = True
            else:
                # The child will not be preserved and the evaluation will continue with the grandchild
                child = grandchild
    return child

# Reduces the vertex count of a list of tree nodes
# Returns a new list of tree nodes
def old_reduce_tree_nodes(tree_nodes, tree_data):
    crit_angle = tree_data.vr_red_angle     # Minimum angle that needs to be preserved
    new_tree_nodes = []     # Return value
    correspondence = {}     # Correspondence table for keeping track of the child indices
    new_tree_nodes.append(tree_nodes[0])    # Copy root
    t1 = time.perf_counter()
    for parent in tree_nodes:
        if parent in new_tree_nodes:    # Check all nodes in tree_nodes that need to be preserved
            list_of_input_tuples = [(child_index, tree_nodes, parent, crit_angle) for child_index in parent.child_indices]
            with mp.Pool(mp.cpu_count()) as pool:    # Foreach of it's children
                children = pool.starmap(_evaluate_branch, list_of_input_tuples)
            for child in children:
                new_tree_nodes.append(child)    # Preserve the chosed child
                correspondence[tree_nodes.index(child)] =  len(new_tree_nodes) - 1  # Add it's new position to the correspondence table
    t2 = time.perf_counter()
    print(f"optimized routine took {t2 - t1} second(s)")
    # Once the reduction itself is done, 
    # rearrange the child indices
    for i, node in enumerate(new_tree_nodes):     # Foreach new node
        for j, c in enumerate(node.child_indices):    # Foreach old index
            new_index = correspondence.get(c, None)     # Search the correspondence table
            while(not new_index):   # If nothing was found in the table (the child was not preserved)
                c = tree_nodes[c].child_indices[0]     # Look at the grandchild
                new_index = correspondence.get(c, None)    # # Search the correspondence table 
                # Repeat until preserved node was found...
            new_tree_nodes[i].child_indices[j] = new_index  # Update the child index
    return new_tree_nodes

# Reduces the vertex count of a list of tree nodes
# Returns a new list of tree nodes
def reduce_tree_nodes(tree_nodes, tree_data):
    crit_angle = tree_data.vr_red_angle     # Minimum angle that needs to be preserved
    parent_indices = [0] * len(tree_nodes)  # List of parent indices
    correspondence = {}     # Correspondences between indices in before and after reduction
    for i, node in enumerate(tree_nodes):   # Setup parent indices
        for c in node.child_indices:
            parent_indices[c] = i
    # Reduction algorithm
    for i, node in enumerate(tree_nodes[1:]): # Foreach node except the root node (since it always remains unchanged) 
        i += 1  # Due to the nature of enumerate i is not matching the elements and must be adjusted
        if len(node.child_indices) == 1:    # Only nodes with exactly one child are candidates for reduction
            parent = tree_nodes[parent_indices[i]]
            child = tree_nodes[node.child_indices[0]]
            vec1 = Vector(node.location - parent.location)  # Vector between node's parent and itself
            vec2 = Vector(child.location - parent.location) # Vector between node's parent and it's child
            angle = vec1.angle(vec2, 0) # Calculate the angle between those two vectors
            if angle < crit_angle:  # If the angle is smaller that specified, the node is not essential
                # Replace parents connection to node with connection to child
                parent.child_indices = [node.child_indices[0] if c == i else c for c in parent.child_indices]
                # Replace child reference to node with reference to parent
                parent_indices[node.child_indices[0]] = parent_indices[i]
                tree_nodes[i] = None    # Mark child as pending removal
    # Update correspondences 
    counter_new = 0
    for counter_old, node in enumerate(tree_nodes):
        if node is not None:
            correspondence[counter_old] = counter_new
            counter_new += 1
    tree_nodes[:] = [node for node in tree_nodes if node is not None]   # Remove None objects from tree_nodes
    for node in tree_nodes:
        node.child_indices = [correspondence[index] for index in node.child_indices]
        
# Evaluates the nodes weight based on the number of recursive children it has
def calculate_weights(tree_nodes):
    n_nodes = len(tree_nodes)   # Total number of tree nodes (relevant for weight factor calculation)
    for node in reversed(tree_nodes):   # For each node (in reversed order)
        for c in node.child_indices:    # Look at each child
            node.weight += tree_nodes[c].weight     # And add the child's (previoulsy calculated) weight to the nodes weight
        node.weight_factor = node.weight / n_nodes  # The weight factor is ratio between all nodes and the nodes that are connected to this particular node

class CreateTree(Operator):
    """Operator that creates a pseudo-random realistic tree"""
    bl_idname = "tbo.create_tree"
    bl_label = "Create Tree"
    bl_description = "Operator that creates a pseudo-random realistic tree"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sel = context.selected_objects
        tree_data = context.scene.tbo_treegen_data
        return ((sel) != []
            and all(obj.type == 'MESH' for obj in sel) 
            and (context.mode == 'OBJECT') 
            and (tree_data.shape_object is not None)
            and (tree_data.shape_object not in sel)
            and (tree_data.sc_d_i > tree_data.sc_d_k or tree_data.sc_d_i == 0)
            and (tree_data.sk_base_radius < tree_data.sc_D)
        )

    def invole(self, context, event):
        return self.execute(context)

    def execute(self, context):
        # Debug information
        os.system("clear")
        f = open("/tmp/log.txt", 'w')
        f.write("Debug Info:\n")
        t_start = time.perf_counter()
        # general-purpose variables
        sel = context.selected_objects  # All objects that shall become a tree
        act = context.active_object
        tree_data = context.scene.tbo_treegen_data

        ### Generate attraction points
        p_attr = get_points_in_object(context, tree_data)

        ### Space colonialization
        # Array that contains a TreeNode object 
        # for each new vertex that will be created later on. It aditionally 
        # stores information like the branching depth, and the parenthood 
        # relationships between nodes.
        all_tree_nodes = []
        # Setup kd-tree of attraction points
        kdt_p_attr = mathutils.kdtree.KDTree(tree_data.n_p_attr)
        for p in range(tree_data.n_p_attr):
            kdt_p_attr.insert(p_attr[p], p)
        kdt_p_attr.balance()
        # Grow stems
        # This is different from the regular iterations since the root nodes 
        # of the tree objects may be far away from the attraction points.
        # In that case extra steps must be taken to grow towards the 
        # attraction points before the regular algorithm can be performed. 
        for tree_obj in sel:
            # Create a new list with one root node
            tree_nodes = []
            tree_nodes.append(Tree_Node(tree_obj.location, tree_obj))
            dist = kdt_p_attr.find(tree_obj.location)[2]
            while (dist > tree_data.sc_d_i * tree_data.sc_D and tree_data.sc_d_i != 0):
                d_i = dist + tree_data.sc_d_i * tree_data.sc_D   # Idjust the attr. point influence so the stem can grow
                iterate_growth(tree_nodes, p_attr, tree_data, d_i)
                dist = kdt_p_attr.find(tree_nodes[-1].location)[2]
            for n in tree_nodes:
                n.child_indices = [(ind + len(all_tree_nodes)) for ind in n.child_indices]
            all_tree_nodes.extend(tree_nodes)
        # Grow the tree crowns
        its = 0
        something_new = True
        while(something_new):
            if((tree_data.sc_n_iter > 0 and its == tree_data.sc_n_iter) or (len(p_attr) == 0)):
                break
            something_new = iterate_growth(all_tree_nodes, p_attr, tree_data)
            its += 1
 
        ### Separate trees
        sorted_trees = {}
        for obj in sel:     # For each tree
            # Create a separate node list
            sorted_trees[obj] = separate_nodes(all_tree_nodes, obj)
        
        ### Calculate depths
        for key in sorted_trees:
            calculate_depths(sorted_trees[key])
        
        ### Calculate weights
        for key in sorted_trees:
            calculate_weights(sorted_trees[key])

        ### Geometry reduction
        if tree_data.pr_enable_reduction:
            for key in sorted_trees:
                reduce_tree_nodes(sorted_trees[key], tree_data)

        ### Generate meshes
        for obj in sel:     # For each tree
            obj_tn = sorted_trees[obj]
            # Calculate vertices in local space 
            verts = [tn.location for tn in obj_tn]
            tf = obj.matrix_world.inverted()
            verts = transform_points(tf, verts)
            # Create bmesh
            bm = bmesh.new()
            # Insert vertices
            for v in verts:
                bm.verts.new(v)
            bm.verts.index_update()
            bm.verts.ensure_lookup_table()
            # Create edges
            for p, n in enumerate(obj_tn):
                for c in n.child_indices:
                    bm.edges.new((bm.verts[p], bm.verts[c]))
            bm.edges.index_update()
            bm.edges.ensure_lookup_table()
            bm.to_mesh(obj.data)
            bm.free()
            if tree_data.pr_enable_skinning:   # If not in preview-mode
                 # Turn the skeletal mesh into one with volume
                skin_skeleton(context, obj, obj_tn, tree_data)

        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        print(f"Finished in {t_finish-t_start} second(s)...")
        return {'FINISHED'}
