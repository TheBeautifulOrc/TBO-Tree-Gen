# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import mathutils
# Debug
import os
import time

from bpy.types import Operator
from mathutils import Vector
from .TreeNodes import TreeNode, TreeNodeContainer
from .SpaceColonialization import grow_trees
from .TreeObjects import TreeObject
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
        all_tree_nodes = grow_trees(tree_data, sel, p_attr)
        ### Separate trees
        sorted_trees = []
        sorted_trees.extend([TreeObject(obj, all_tree_nodes.separate_by_object(obj), tree_data) for obj in sel])
        
        for tree in sorted_trees:
            ### Calculate weights
            tree.nodes.calculate_weights()
            ### Reduce unnecessary nodes 
            tree.nodes.reduce_nodes(tree_data.nr_max_angle)
            ### Generate mesh
            if not tree_data.pr_enable_skinning:
                tree.generate_skeltal_mesh() # Generate skeleton
            # If not in preview-mode create mesh with volume
            else:
                tree.generate_mesh()
            
        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        self.report({'INFO'}, f"Created {len(sel)} tree(s) in {round(t_finish-t_start, 3)} second(s)")
        return {'FINISHED'}
