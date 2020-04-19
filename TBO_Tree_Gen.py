# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

bl_info = {
    "name" : "TBO's TreeGenerator",
    "author" : "TheBeautifulOrc",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 8, 0),
    "location" : "3DView",
    "warning" : "",
    "category" : "Add Mesh"
}

import bpy
import mathutils
import bmesh
import numpy as np
import os
import math
import time
import concurrent.futures

from bpy.types import (Operator, Panel, PropertyGroup)
from bpy.props import (IntProperty, BoolProperty, StringProperty, FloatProperty, EnumProperty, PointerProperty)
from mathutils import Vector
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List
from multiprocessing import Process

class TreeProperties(PropertyGroup):
    def shape_object_poll(self, obj):
        return ((obj.type == 'MESH') 
            and (obj not in bpy.context.selected_objects)
            and (obj.name in bpy.data.objects)
        )

    pr_enable_reduction : BoolProperty(
        name="Enable Vertex Reduction",
        description="Should the vertex reduction be executed?",
        default=True
    )
    pr_enable_skinning : BoolProperty(
        name="Enable Skinning",
        description="Should the skin modifier be applied?",
        default=True
    )
    seed : IntProperty(
        name="Seed",
        description="Seed for tree randomization",
        default=0
    )
    shape_object : PointerProperty(
        type=bpy.types.Object,
        name="Shape Object",
        description="Object that defines the trees overall shape",
        poll=shape_object_poll
    )
    use_shape_modifiers : BoolProperty(
        name="Evaluate Modifiers", 
        description="Should the shape objects modifiers be evaluated?",
        default=True
    )
    n_p_attr : IntProperty(
        name="Number of Attraction Points",
        description="Number of attraction points used to generate the tree model",
        default=1000,
        min=1
    )
    emit_p_attr : EnumProperty(
        items=[('VOLUME', "Volume", "Points get scattered across the objects volume"),
            ('FACE', "Surface", "Points get scattered across the objects surface"),
            ('VERT', "Vertices", "Points get placed on the objects vertices")],
        name="Attraction Point Emitter",
        description="Determines where the attracion points get emitted from",
        default='VOLUME'
    )
    dist_p_attr : EnumProperty(
        items=[('RAND', "Random", "Random distribution"),
            ('JIT', "Jittered", "Distribution on a grid with random variation")],
        name="Attraction Point Distribution",
        description="Method of distributing the attraction points",
        default='RAND'
    )
    even_dist_p_attr : BoolProperty(
        name="Even Distribution",
        description="Should attraction points be distributed evenly?",
        default=True
    )
    sc_D : FloatProperty(
        name="Node Distance",
        description="Distance in between adjacent nodes",
        default=0.1,
        min=0.01, 
        max=1.0,
        unit='LENGTH'
    )
    sc_d_i : IntProperty(
        name="Influence Factor",
        description="Radius (in Node Distances) in which an attraction point can influence the growing tree ('0' means the radius is infinite)",
        default=10,
        min=0
    )
    sc_d_k : IntProperty(
        name="Kill Factor",
        description="Distance at which an attraction point gets removed if the tree grows too close",
        default=2,
        min=1
    )
    sc_n_iter : IntProperty(
        name="Max Iterations",
        description="Maximum amount of iterations the space colonialization algorithm may go through ('0' for unlimited iterations)",
        default=1000,
        min=0
    )
    vr_red_angle : FloatProperty(
        name="Reduction Angle",
        description="Smallest angle that can't be reduced anymore",
        default=math.radians(5.0),
        soft_min=0.1,
        min=0.0,
        max=math.radians(90.0),
        subtype='ANGLE',
        unit='ROTATION'
    )
    sk_base_radius : FloatProperty(
        name="Base Radius",
        description="Radius at the very base of the tree trunk",
        default=0.5,
        min=0.0,
        soft_min=0.01,
        unit='LENGTH'
    )
    sk_min_radius : FloatProperty(
        name="Minimum Radius",
        description="Minimum radius of the branches",
        default=0.01,
        min=0.0,
        unit='LENGTH'
    )
    sk_smoothing : FloatProperty(
        name="Branch Smoothing",
        description="Determines how much complex geometry around the branches should be smoothed",
        default=1.0,
        max=1.0,
        min=0.0
    )

@dataclass
class Tree_Node:
    location : Vector
    parent_object : bpy.types.Object
    depth : int = 0
    weight : int = 1
    weight_factor : float = 0.0
    child_indices : List[int] = field(default_factory=list)

class CreateTree(Operator):
    """Operator that creates a pseudo-random realistic tree"""
    bl_idname = "tbo.create_tree"
    bl_label = "Create Tree"
    bl_description = "Operator that creates a pseudo-random realistic tree"
    bl_options = {'REGISTER', 'UNDO'}

    # Transfomrs a list of points according to the given 
    # 4x4 transformation matrix using numpy.
    # Returns the transformed vertices as a list. 
    def transform_points(self, transf_matr, points):
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
    def get_points_in_object(self, context, tree_data, temp_name="temp_part_sys"):
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
    def iterate_growth(self, growth_nodes, p_attr, tree_data, d_i_override=None):
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
    def calculate_depths(self, tree_nodes):
        for tn in tree_nodes:   # For each tree node 
            if len(tn.child_indices) > 1:   # If the node is a branching point
                child_depth = tn.depth + 1  # The child branches shlould be of higher depth
            else:   # If there's only one or no child
                child_depth = tn.depth  # The child is a continuation of the current branch and should therefore have the same depth
            for child in tn.child_indices:
                tree_nodes[child].depth = child_depth

    # Seperates a list of mixed nodes.
    # Returns a list of all nodes that have obj as parent.
    def separate_nodes(self, mixed_nodes, obj):
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
    def skin_skeleton(self, context, obj, tree_nodes, tree_data, temp_name="temp_skin_mod"):
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

    # Reduces the vertex count of a list of tree nodes
    # Returns a new list of tree nodes
    def reduce_tree_nodes(self, tree_nodes, tree_data):
        crit_angle = tree_data.vr_red_angle     # Minimum angle that needs to be preserved
        new_tree_nodes = []     # Return value
        correspondence = {}     # Correspondence table for keeping track of the child indices
        new_tree_nodes.append(tree_nodes[0])    # Copy root
        for parent in tree_nodes:
            if parent in new_tree_nodes:    # Check all nodes in tree_nodes that need to be preserved
                for c in parent.child_indices:    # Foreach of it's children
                    child = tree_nodes[c]   # Get the child node
                    something_to_preserve = False
                    while(not something_to_preserve):
                        if len(child.child_indices) != 1:   # If the child, we are looking at, is a branch or endpoint, it must be preserved and the routine can end
                            something_to_preserve = True                         
                        else:   # Else we have to further evaluate if the child should be preserved
                            grandchild = tree_nodes[child.child_indices[0]]     # Take a look at the child's child
                            # Calculate the angle between the line segments [parent;child] and [parent;grandchild]
                            vec1 = Vector(parent.location - child.location)
                            vec2 = Vector(parent.location - grandchild.location)
                            angle = vec1.angle(vec2, 0)
                            if angle >= crit_angle:     # If the angle is larger than specified by the user, the child must be preserved
                                something_to_preserve = True
                            else:
                                # The child will not be preserved and the evaluation will continue with the grandchild
                                child = grandchild
                    new_tree_nodes.append(child)    # Preserve the chosed child
                    correspondence[tree_nodes.index(child)] =  len(new_tree_nodes) - 1  # Add it's new position to the correspondence table
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

    # Evaluates the nodes weight based on the number of recursive children it has
    def calculate_weights(self, tree_nodes):
        n_nodes = len(tree_nodes)   # Total number of tree nodes (relevant for weight factor calculation)
        for node in reversed(tree_nodes):   # For each node (in reversed order)
            for c in node.child_indices:    # Look at each child
                node.weight += tree_nodes[c].weight     # And add the child's (previoulsy calculated) weight to the nodes weight
            node.weight_factor = node.weight / n_nodes  # The weight factor is ratio between all nodes and the nodes that are connected to this particular node

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
        p_attr = self.get_points_in_object(context, tree_data)

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
                self.iterate_growth(tree_nodes, p_attr, tree_data, d_i)
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
            something_new = self.iterate_growth(all_tree_nodes, p_attr, tree_data)
            its += 1
 
        ### Separate trees
        sorted_trees = {}
        for obj in sel:     # For each tree
            # Create a separate node list
            sorted_trees[obj] = self.separate_nodes(all_tree_nodes, obj)
        
        ### Calculate depths
        for key in sorted_trees:
            self.calculate_depths(sorted_trees[key])
        
        ### Calculate weights
        for key in sorted_trees:
            self.calculate_weights(sorted_trees[key])

        #f.write("Before\n")
        #for key in sorted_trees:
        #    for i, node in enumerate(sorted_trees[key]):
        #        f.write(str(i) + " " + str(node) + "\n")

        ### Geometry reduction
        if tree_data.pr_enable_reduction:
            for key in sorted_trees:
                sorted_trees[key] = self.reduce_tree_nodes(sorted_trees[key], tree_data)

        #f.write("After\n")
        #for key in sorted_trees:
        #    for i, node in enumerate(sorted_trees[key]):
        #        f.write(str(i) + " " + str(node) + "\n")

        ### Generate meshes
        for obj in sel:     # For each tree
            obj_tn = sorted_trees[obj]
            # Calculate vertices in local space 
            verts = [tn.location for tn in obj_tn]
            tf = obj.matrix_world.inverted()
            verts = self.transform_points(tf, verts)
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
                self.skin_skeleton(context, obj, obj_tn, tree_data)

        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        print(f"Finished in {t_finish-t_start} second(s)...")
        return {'FINISHED'}

class PanelTemplate:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Create"
    bl_options = {'DEFAULT_CLOSED'}

class MainPanel(PanelTemplate, Panel):
    bl_idname = "OBJECT_PT_tbo_treegen_main"
    bl_label = "TBO's TreeGenerator"

    @classmethod
    def poll(cls, context):
        sel = context.selected_objects
        return ((sel) != [] 
            and all(obj.type == 'MESH' for obj in sel) 
            and (context.mode == 'OBJECT')
        )

    def draw(self, context):
        layout = self.layout
        layout.operator("tbo.create_tree")

class PRSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_pr"
    bl_label = "Preview Settings"

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        grid.label(text="Vertex Reduction")
        grid.prop(tree_data, "pr_enable_reduction", text="")
        grid.label(text="Skin Modifier")
        grid.prop(tree_data, "pr_enable_skinning", text="")

class APSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_ap"
    bl_label = "Attraction Points"

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data
        separation_factor = 2.0

        grid = layout.grid_flow(row_major=True, columns=2)
        grid.label(text="Shape Object")
        grid.prop(tree_data, "shape_object", text="")
        grid.label(text="Use Modifiers")
        grid.prop(tree_data, "use_shape_modifiers", text="")
        grid.separator(factor=separation_factor)
        grid.separator(factor=separation_factor)
        grid.label(text="Number")
        grid.prop(tree_data, "n_p_attr", text="")
        grid.label(text="Seed")
        grid.prop(tree_data, "seed", text="")
        grid.label(text="Emit from")
        grid.prop(tree_data, "emit_p_attr", text="")
        grid.label(text="Distribution")
        grid.prop(tree_data, "dist_p_attr", text="")
        grid.label(text="Even distribution")
        grid.prop(tree_data, "even_dist_p_attr", text="")

class SCSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_sc"
    bl_label = "Space Colonialization"

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data
        separation_factor = 2.0

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Node Distance")
        grid.prop(tree_data, "sc_D", text="")
        grid.separator(factor=separation_factor)
        grid.separator(factor=separation_factor)
        grid.label(text="Influence Radius")
        grid.prop(tree_data, "sc_d_i", text="")
        grid.label(text="Kill Distance")
        grid.prop(tree_data, "sc_d_k", text="")
        grid.separator(factor=separation_factor)
        grid.separator(factor=separation_factor)
        grid.label(text="Max Iterations")
        grid.prop(tree_data, "sc_n_iter", text="")

class VRSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_vr"
    bl_label = "Vertex Reduction"

    @classmethod
    def poll(cls, context):
        tree_data = context.scene.tbo_treegen_data
        return tree_data.pr_enable_reduction
    
    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Reduciton Angle")
        grid.prop(tree_data, "vr_red_angle", text="")

class SKSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_sk"
    bl_label = "Skinning"

    @classmethod
    def poll(cls, context):
        tree_data = context.scene.tbo_treegen_data
        return tree_data.pr_enable_skinning

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Base Radius")
        grid.prop(tree_data, "sk_base_radius", text="")
        grid.label(text="Minimum Radius")
        grid.prop(tree_data, "sk_min_radius", text="")
        grid.label(text="Branch Smoothing")
        grid.prop(tree_data, "sk_smoothing", text="")

classes = (
    TreeProperties, 
    CreateTree, 
    MainPanel, 
    PRSubPanel,
    APSubPanel,
    SCSubPanel,
    VRSubPanel,
    SKSubPanel
)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)
    bpy.types.Scene.tbo_treegen_data = PointerProperty(type=TreeProperties)

def unregister():
    for cl in reversed(classes):
        bpy.utils.unregister_class(cl)
    del bpy.types.Scene.tbo_treegen_data
