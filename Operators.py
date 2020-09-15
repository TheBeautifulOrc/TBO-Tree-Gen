# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import mathutils
# Debug
import os
import time

from bpy.types import Operator
from mathutils import Vector
from .TreeNodes import Tree_Node, Tree_Node_Container
from .TreeObjects import Tree_Object
from .Utility import get_points_in_object

class CreateTree(Operator):
    """Operator that creates a pseudo-random, realistic looking tree"""
    bl_idname = "tbo.create_tree"
    bl_label = "Create Tree"
    bl_description = "Operator that creates a pseudo-random, realistic looking tree"
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
        )

    def invole(self, context, event):
        return self.execute(context)

    def execute(self, context):
        # Debug information
        # TODO: Remove debugging overhead
        try:
            os.system("clear")  # Will fail on Windows 
        except:
            pass
        f = open("/tmp/log.txt", 'w')
        f.write("Debug Info:\n")
        t_start = time.perf_counter()
        
        # General purpose variables
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
        all_tree_nodes = Tree_Node_Container()
        # Setup kd-tree of attraction points
        kdt_p_attr = mathutils.kdtree.KDTree(tree_data.n_p_attr)
        for i, p in enumerate(p_attr):
            kdt_p_attr.insert(p, i)
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
        sorted_trees = []
        sorted_trees.extend([Tree_Object(obj, all_tree_nodes.separate_by_object(obj), tree_data) for obj in sel])
        
        for tree in sorted_trees:
            ### Calculate weights
            tree.nodes.calculate_weights()
            ### Geometry reduction
            if tree_data.pr_enable_reduction:
                tree.nodes.reduce_tree_nodes(tree_data)
            ### Generate mesh
            if not tree_data.pr_enable_skinning:
                tree.generate_skeltal_mesh() # Generate skeleton
            # If not in preview-mode create mesh with volume
            else:
                t1 = time.perf_counter()
                tree.generate_mesh_ji_liu_wang()
                t2 = time.perf_counter()
                print(t2-t1)
            
        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        self.report({'INFO'}, f"Created {len(sel)} tree(s) in {round(t_finish-t_start, 3)} second(s)...")
        return {'FINISHED'}
