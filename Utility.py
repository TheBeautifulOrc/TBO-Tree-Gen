import bpy
import bmesh
import numpy as np
import math 

from .TreeNodes import Tree_Node, Tree_Node_Container
from mathutils import Vector
from dataclasses import dataclass

la = np.linalg
mask = np.ma

# Transforms a list of points according to the given 
# 4x4 transformation matrix using numpy.
# Returns the transformed vertices as a list. 
def transform_points(transf_matr, points):
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
def get_points_in_object(context, tree_data, temp_name="temp_part_sys"):
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
