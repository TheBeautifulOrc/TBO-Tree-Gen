# Copyright (C) 2019-2021 Luai "TheBeautifulOrc" Malek

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
from bpy.types import (Operator, Panel, PropertyGroup)
from .TreeProperties import TreeProperties

_separation_factor = 2.0

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
        grid.label(text="Skeletons Only")
        grid.prop(tree_data, "pr_skeletons_only", text="")

class APSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_ap"
    bl_label = "Attraction Points"

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        grid.label(text="Shape Object")
        grid.prop(tree_data, "shape_object", text="")
        grid.label(text="Use Modifiers")
        grid.prop(tree_data, "use_shape_modifiers", text="")
        grid.separator(factor=_separation_factor)
        grid.separator(factor=_separation_factor)
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

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Node Distance")
        grid.prop(tree_data, "sc_D", text="")
        grid.separator(factor=_separation_factor)
        grid.separator(factor=_separation_factor)
        grid.label(text="Influence Radius")
        grid.prop(tree_data, "sc_d_i", text="")
        grid.label(text="Kill Distance")
        grid.prop(tree_data, "sc_d_k", text="")
        grid.separator(factor=_separation_factor)
        grid.separator(factor=_separation_factor)
        grid.label(text="Max Iterations")
        grid.prop(tree_data, "sc_n_iter", text="")

class NRSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_vr"
    bl_label = "Node Reduction"
    
    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Reduction Angle")
        grid.prop(tree_data, "nr_max_angle", text="")

class SKSubPanel(PanelTemplate, Panel):
    bl_parent_id = "OBJECT_PT_tbo_treegen_main"
    bl_idname = "OBJECT_PT_tbo_treegen_sk"
    bl_label = "Skinning"

    @classmethod
    def poll(cls, context):
        tree_data = context.scene.tbo_treegen_data
        return not tree_data.pr_skeletons_only

    def draw(self, context):
        layout = self.layout
        tree_data = context.scene.tbo_treegen_data

        grid = layout.grid_flow(row_major=True, columns=2)
        
        grid.label(text="Base Radius")
        grid.prop(tree_data, "sk_base_radius", text="")
        grid.label(text="Minimum Radius")
        grid.prop(tree_data, "sk_min_radius", text="")
        grid.separator(factor=_separation_factor)
        grid.separator(factor=_separation_factor)
        grid.label(text="Loop Distance")
        grid.prop(tree_data, "sk_loop_distance", text="")
        grid.separator(factor=_separation_factor)
        grid.separator(factor=_separation_factor)
        grid.label(text="Interpolation Mode")
        grid.prop(tree_data, "sk_interpolation_mode", text="")