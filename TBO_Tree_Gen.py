bl_info = {
    "name" : "TBO's TreeGenerator",
    "author" : "TheBeautifulOrc",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 3),
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

from bpy.types import (Operator, Panel, PropertyGroup)
from bpy.props import (IntProperty, BoolProperty, StringProperty, FloatProperty, EnumProperty, PointerProperty)
from mathutils import Vector
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List

class TreeProperties(PropertyGroup):
    seed : IntProperty(
        name="Seed",
        description="Seed for tree randomization",
        default=0
    )
    shape_object : PointerProperty(
        type=bpy.types.Object,
        name="Shape Object",
        description="Object that defines the trees overall shape"
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
        max=1.0
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
    sc_branch_depth : IntProperty(
        name="Max Branch Depth",
        description="Maximum branching depth the tree can develop during growth ('0' for unlimited branching)",
        default=0,
        min=0
    )
    sc_n_iter : IntProperty(
        name="Max Iterations",
        description="Maximum amount of iterations the space colonialization algorithm may go through ('0' for unlimited iterations)",
        default=1000,
        min=0
    )
    simp_angle : FloatProperty(
        name="Stepping Angle",
        description="This angle determines how much the mesh may be simplified (0.0 means no simplification)",
        default=5.0,
        min=0.0,
        max=45.0
    )

@dataclass
class Tree_Node:
    location : Vector
    parent_object : bpy.types.Object
    depth : int = 0
    child_indices : List[int] = field(default_factory=list)

class CreateTree(Operator):
    """Operator that creates a pseudo-random realistic tree"""
    bl_idname = "tbo.create_tree"
    bl_label = "Create Tree"
    bl_description = "Operator that creates a pseudo-random realistic tree"
    bl_options = {'REGISTER', 'UNDO'}

    def get_mesh_vertices(self, obj, arr=None):
        me = obj.to_mesh()
        l = len(me.vertices)
        if arr is None:
            arr = np.zeros(l*3 , dtype=np.float64)
        else:
            arr.resize(l*3)
        me.vertices.foreach_get('co', arr.ravel())
        return arr.reshape(l,3)

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

    def iterate_growth(self, growth_nodes, p_attr, d_i, d_k, D, max_depth):
        # Create kd-tree of all nodes
        n_nodes = len(growth_nodes)
        kdt_nodes = mathutils.kdtree.KDTree(n_nodes)
        for i, node in enumerate(growth_nodes):
            kdt_nodes.insert(node.location, i)
        kdt_nodes.balance()
        # Return values
        kill = []
        new_nodes = []
        # Build correspondence table between nodes and attr. points
        corr = defaultdict(list)
        for p in p_attr:
            vec, ind, dist = kdt_nodes.find(p)
            # Kill overgrown attr. points
            if dist <= d_k:
                kill.append(p)
            # Else assign close attr. points to nodes
            elif (dist <= d_i or d_i == 0) and (max_depth == 0 or growth_nodes[ind].depth < max_depth):   # Only if they're not already at max depth
                corr[ind].append(p)
        # If the node has close attr. points
        # grow a child node
        for i, key in enumerate(corr):
            parent_node = growth_nodes[key]
            loc = parent_node.location
            n_vec = Vector((0.0,0.0,0.0)) 
            for p in corr[key]:
                n_vec += (p - loc).normalized()
            if n_vec.length < math.sqrt(2):
                n_vec = (corr[key][0] - loc).normalized()
            # Create child node:
            # Evaluate child nodes depth
            parent_node.child_indices.append(n_nodes + i)
            if(len(parent_node.child_indices) > 1):
                c_depth = parent_node.depth + 1     # One greater than parents depth if the parent is a branching point
                if(len(parent_node.child_indices) == 2):    # Special case: Parent node just became a branching node
                    for c in parent_node.child_indices[:-1]:    # Iterate through all previous nodes
                        growth_nodes[c].depth = c_depth     # And increase their depth
            else:
                c_depth = parent_node.depth
            new_nodes.append(Tree_Node((loc + D * n_vec.normalized()), parent_node.parent_object, c_depth))
        growth_nodes.extend(new_nodes)
        for key in corr:
            n = growth_nodes[key]
            if len(n.child_indices) > 1:
                for c in n.child_indices:
                    growth_nodes[c].depth = n.depth + 1

        for p in kill:
            p_attr.remove(p)
        return (len(corr) > 0)

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

    @classmethod
    def poll(cls, context):
        sel = context.selected_objects
        tree_data = context.scene.tbo_treegen_data
        return ((sel) != []
            and all(obj.type == 'MESH' for obj in sel) 
            and (context.mode == 'OBJECT') 
            and (tree_data.shape_object is not None) 
            and (tree_data.shape_object.type == 'MESH')
            and (tree_data.shape_object not in sel)
            and (tree_data.sc_d_i > tree_data.sc_d_k or tree_data.sc_d_i == 0)
        )

    def invole(self, context, event):
        return self.execute(context)

    def execute(self, context):
        os.system("clear")
        sel = context.selected_objects  # All objects that shall become a tree
        tree_data = context.scene.tbo_treegen_data
        f = open('log.txt', 'w')

        ### Generate attraction points
        p_attr = self.get_points_in_object(context, tree_data)

        ### Space colonialization
        # Array that contains a TreeNode object 
        # for each new vertex that will be created later on. It aditionally 
        # stores information like the branching depth, and the parenthood 
        # relationships between nodes.
        all_tree_nodes = []   # Important!

        # Setup kd-tree of attraction points
        kdt_p_attr = mathutils.kdtree.KDTree(tree_data.n_p_attr)
        for p in range(tree_data.n_p_attr):
            kdt_p_attr.insert(p_attr[p], p)
        kdt_p_attr.balance()

        # Grow stems:
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
                self.iterate_growth(tree_nodes, p_attr, d_i, tree_data.sc_d_k * tree_data.sc_D, tree_data.sc_D, tree_data.sc_branch_depth)
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
            something_new = self.iterate_growth(all_tree_nodes, p_attr, tree_data.sc_d_i * tree_data.sc_D, tree_data.sc_d_k * tree_data.sc_D, tree_data.sc_D, tree_data.sc_branch_depth)
            its += 1
        # Generate meshes:
        for obj in sel:     # For each tree
            # Create a separate node list
            obj_tn  = self.separate_nodes(all_tree_nodes, obj)
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
        grid.label(text="Max Branch Depth")
        grid.prop(tree_data, "sc_branch_depth", text="")
        grid.label(text="Max Iterations")
        grid.prop(tree_data, "sc_n_iter", text="")

classes = (
    TreeProperties, 
    CreateTree, 
    MainPanel, 
    APSubPanel,
    SCSubPanel
)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)
    bpy.types.Scene.tbo_treegen_data = PointerProperty(type=TreeProperties)

def unregister():
    for cl in reversed(classes):
        bpy.utils.unregister_class(cl)
    del bpy.types.Scene.tbo_treegen_data
