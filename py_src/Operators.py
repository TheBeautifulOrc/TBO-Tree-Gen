# Copyright (C) 2019-2020 Luai "TheBeautifulOrc" Malek

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# pylint: disable=relative-beyond-top-level, 

import bpy
import mathutils
# Debug
import os
import time
import sys

from bpy.types import Operator
from mathutils import Vector
import numpy as np
from .TreeProperties import TreeProperties
from .TreeObjects import TreeObject
from .Utility import get_points_in_object
from ..cpp_bin.TreeGenModule import TreeNode, TreeNodeContainer, grow_nodes, separate_by_id, reduce_nodes, calculate_weights
from .MeshGenration import genrate_skeletal_mesh, generate_mesh

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

    def invoke(self, context, event):
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
        tree_data : TreeProperties = context.scene.tbo_treegen_data

        ### Generate attraction points
        p_attr = get_points_in_object(context, tree_data)

        ### Space colonialization
        all_tree_nodes = TreeNodeContainer()
        all_tree_nodes.extend([TreeNode(obj.location, id(obj)) for obj in sel])
        p_attr = np.asfortranarray(p_attr)
        grow_nodes(all_tree_nodes, p_attr, tree_data.sc_D, tree_data.sc_d_i, tree_data.sc_d_k, tree_data.sc_n_iter)
        
        ### Separate trees
        for obj in sel:
            tnc = separate_by_id(all_tree_nodes, id(obj))
            ### Calculate weights
            calculate_weights(tnc)
            ### Reduce unnecessary nodes 
            reduce_nodes(tnc, tree_data.nr_max_angle)
            ### Generate mesh
            if tree_data.pr_skeletons_only:
                genrate_skeletal_mesh(obj, tnc)
            # If not in preview-mode create mesh with volume
            else:
                generate_mesh(obj, tnc, tree_data)
            
        # Reset active object
        context.view_layer.objects.active = act
        
        f.close()
        t_finish = time.perf_counter()
        self.report({'INFO'}, f"Created {len(sel)} tree(s) in {round(t_finish-t_start, 3)} second(s)")
        return {'FINISHED'}
