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
from .TreeNodes import Tree_Node, Tree_Node_Container

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
   
    # Takes in an object and a Tree_Node_Container to generate 
    # the objects skeletal mesh using bmesh
    def generate_skeltal_mesh(self, obj, obj_tn):
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
                if p == c:
                    print(obj_tn)
                bm.edges.new((bm.verts[p], bm.verts[c]))
        bm.edges.index_update()
        bm.edges.ensure_lookup_table()
        bm.to_mesh(obj.data)
        bm.free()

    # Adds and applies a skin modifier to the skeletal mesh
    def skin_skeleton(self, context, obj, tree_nodes, tree_data, temp_name="temp_skin_mod"):
        context_override = context.copy()   # New temporary context
        context_override['active_object'] = obj    # The passed object is viewed as selected
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
        bpy.ops.object.modifier_apply(context_override, modifier=temp_name)

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
        all_tree_nodes = Tree_Node_Container()
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
            tree_nodes = Tree_Node_Container()
            tree_nodes.append(Tree_Node(tree_obj.location, tree_obj))
            dist = kdt_p_attr.find(tree_obj.location)[2]
            while (dist > tree_data.sc_d_i * tree_data.sc_D and tree_data.sc_d_i != 0):
                d_i = dist + tree_data.sc_d_i * tree_data.sc_D   # Adjust the attr. point influence so the stem can grow
                tree_nodes.iterate_growth(p_attr, tree_data, d_i)
                dist = kdt_p_attr.find(tree_nodes[-1].location)[2]
            for n in tree_nodes:
                n.child_indices = [(ind + len(all_tree_nodes)) for ind in n.child_indices]
            all_tree_nodes.extend(tree_nodes)
        # Grow the tree crowns
        its = 0
        something_new = True
        while(something_new):
            if((tree_data.sc_n_iter > 0 and its == tree_data.sc_n_iter) or (len(p_attr) == 0)):
                something_new = False
            else:
                something_new = all_tree_nodes.iterate_growth(p_attr, tree_data)
                its += 1
        
        ### Separate trees
        sorted_trees = {}
        for obj in sel:     # For each tree
            # Create a separate node list
            sorted_trees[obj] = all_tree_nodes.separate_by_object(obj)
        
        for obj in sel:    
            obj_tn = sorted_trees[obj]  # For each separate list of tree nodes 
            ### Calculate weights
            obj_tn.calculate_weights()
            ### Geometry reduction
            if tree_data.pr_enable_reduction:
                obj_tn.reduce_tree_nodes(tree_data)
            ### Generate mesh
            # If not in preview-mode create mesh with volume
            if tree_data.pr_enable_skinning:   
                # To avoid glitches with the skin modifier 
                # each branch is skinned individually
                branches = obj_tn.separated_branches()  # Get separeted branches 
                for i, branch in enumerate(branches):
                    # Create a new temporary  object
                    # based on a temporary mesh
                    temp_mesh = bpy.data.meshes.new(f"temp_mesh")
                    # If this is the first branch, the base object can be used instead
                    temp_obj = obj if i == 0 else bpy.data.objects.new(f"temp_obj", temp_mesh)
                    # Generate skeletal mesh of the branch
                    self.generate_skeltal_mesh(temp_obj, branch)
                    # Turn the skeletal mesh into one with volume
                    self.skin_skeleton(context, temp_obj, branch, tree_data)
                    if i != 0:
                        context_override = context.copy()
                        context_override['active_object'] = obj
                        temp_bool_mod = obj.modifiers.new(name="temp_bool", type='BOOLEAN')
                        temp_bool_mod.operation = 'UNION'
                        temp_bool_mod.object = temp_obj
                        bpy.ops.object.modifier_apply(modifier="temp_bool")
                        bpy.data.objects.remove(temp_obj)
                    bpy.data.meshes.remove(temp_mesh)
            else:   # If in preview-mode only create a skeleton
                self.generate_skeltal_mesh(obj, obj_tn)
                                
        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        print(f"Finished in {t_finish-t_start} second(s)...")
        return {'FINISHED'}
