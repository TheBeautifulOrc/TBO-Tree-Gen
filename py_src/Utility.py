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
from mathutils import Vector
import numpy as np

def get_points_in_object(context, tree_data, temp_name="temp_part_sys"):
    """
    Returns points within mesh.
    
    Generates points in/on an object using a Blender particle system.
    
    Keyword arguments:
        context : bpy.context 
            Context variable of the current Blender session
        tree_data : TreeProperties 
            List of all properties of the tree that is being worked on
        temp_part_sys : string
            Name for the temporary particle system that is being used
            
    Return value: 
        List of (Blender) mathutils.Vector
    """
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
    arr = np_arr.tolist()
    arr = [Vector(e) for e in arr]
    return arr