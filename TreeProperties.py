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

import bpy 
import math
from bpy.props import PointerProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty
from bpy.types import PropertyGroup

class TreeProperties(PropertyGroup):
    """
    Blender property used for all tree settings.
    
    A custom Blender property that holds all user settings 
    controlling the creation of a new tree. 
    It is individually stored within each .blend-file.
    """
    def shape_object_poll(self, obj):
        """Polling function that checks whether the given shape object is valid."""
        return ((obj.type == 'MESH') 
            and (obj.name in bpy.data.objects)
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
        description="Determines where the attraction points get emitted from",
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
    nr_max_angle : FloatProperty(   
        name="Reduction Angle",
        description="Largest angle that will be considered a straight line and thus be reduced",
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
    sk_loop_distance : FloatProperty(   
        name="Loop Distance",
        description="Preferred distance between edge loops",
        default=0.1,
        min=0.000001,
        unit='LENGTH'
    )
    sk_interpolation_mode : EnumProperty(   
        items=[('LIN', "Linear", "Linear Interpolation"),
               ('SPL', "Spline", "Spline Interpolation")],
        name="Interpolation Mode",
        description="Method for interpolating positions in between nodes",
        default='LIN'
    )