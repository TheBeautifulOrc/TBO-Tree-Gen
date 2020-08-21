# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
from bpy.types import (Operator, Panel, PropertyGroup)
from .TreeProperties import TreeProperties

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
        grid.label(text="Tree Skinning")
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
        
        grid.label(text="Reduction Angle")
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
