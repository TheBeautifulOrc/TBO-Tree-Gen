# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

bl_info = {
    "name" : "TBO_TreeGenerator",
    "author" : "TheBeautifulOrc",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 8, 2),
    "location" : "3DView",
    "warning" : "",
    "category" : "Add Mesh" 
}

import bpy
import math

from bpy.props import (PointerProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty)
from bpy.types import PropertyGroup

if "bpy" in locals():
    import importlib
    from . import Operators
    importlib.reload(Operators)
    from . import Panels
    importlib.reload(Panels)
    from . import TreeNodes
    importlib.reload(TreeNodes)
    from . import Utility
    importlib.reload(Utility)
else:
    from . import Operators
    from . import Panels

class TreeProperties(PropertyGroup):
    def shape_object_poll(self, obj):
        return ((obj.type == 'MESH') 
            and (obj not in bpy.context.selected_objects)
            and (obj.name in bpy.data.objects)
        )
        
    def sk_profile_curve_poll(self, obj):
        return ((obj.type == 'CURVE') 
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
        default=0.25,
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
        default=0.15,
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

classes = (
    TreeProperties, 
    Operators.CreateTree, 
    Panels.MainPanel, 
    Panels.PRSubPanel,
    Panels.APSubPanel,
    Panels.SCSubPanel,
    Panels.VRSubPanel,
    Panels.SKSubPanel
)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)
    bpy.types.Scene.tbo_treegen_data = PointerProperty(type=TreeProperties)

def unregister():
    for cl in reversed(classes):
        bpy.utils.unregister_class(cl)
    del bpy.types.Scene.tbo_treegen_data
