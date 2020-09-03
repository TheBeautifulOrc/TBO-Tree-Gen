# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
import numpy as np
import math 

from .TreeNodes import Tree_Node, Tree_Node_Container
from mathutils import Vector
from dataclasses import dataclass

la = np.linalg
mask = np.ma

def transform_points(transf_matr, points):
    """
    Transforms points according to the given 4x4 transformation matrix.
    
    Keyword arguments:
        transf_matr : mathutils.Matrix [4x4]
            Transformation matrix
        points : list(mathutils.Vector)
            Points that shall be transformed
        
    Return value:
        List of transformed points. 
    """
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

def get_first_last_inds(arr):
    """
    Return indices of the first and the last non-nan entry from a numpy array.
    
    Keyword arguments:
        arr : numpy.array
            The array in question
    """
    non_nan_indices = np.transpose(np.where(~np.isnan(arr)))
    splitter = np.flatnonzero(np.diff(non_nan_indices[:,0])) + 1
    separated_indices = np.array_split(non_nan_indices, splitter)
    first_indices_incomplete = np.array([b[0] for b in separated_indices])
    last_indices_incomplete = np.array([b[-1] for b in separated_indices])
    first_indices = np.full((arr.shape[0], 2), [-1,0], dtype=np.int)
    last_indices = np.full_like(first_indices, [-1,0])
    for elem_f, elem_l in zip(first_indices_incomplete, last_indices_incomplete):
        first_indices[elem_f[0]] = elem_f
        last_indices[elem_l[0]] = elem_l
    return first_indices, last_indices

def quick_hull(points):
    """
    Generates convex hulls.
    
    Generates convex hulls around all given sets of points. 
    
    Keyword arguments: 
        points : numpy.array 
            Numpy array with structe [set, point, members/index (4D)]
    """ 
    def dist_to_line(line_points, test_points):
        """
        Calculates the distance between lines and a sets of points.
        
        Calculates the minimum distance between lines and sets of points. 
        The lines are defined by two points each that are part of the respective line.
        
        Keyword arguments:
            line_points : np.array [set, point, member/index(4D)]
                Points that define the line
            test_points : np.array [set, point, member/index(4D)]
                Points whose distance to the line shall be calculated

        Return value:
            Numpy array containing the distance of each point to it's respective line.
        """
        line_points = line_points[:,:,:-1]
        test_points = test_points[:,:,:-1]
        n = line_points[:,1] - line_points[:,0]
        n /= la.norm(n, 2, -1, keepdims=True)
        diff = test_points - np.repeat(line_points[:,0], test_points.shape[1], axis=0).reshape(test_points.shape)
        term_1 = np.einsum('spm,sm->sp',diff,n)
        dist = np.abs(diff - np.einsum('sp,sm->spm',term_1,n))
        dist = la.norm(dist, 2, -1, keepdims=True)
        return dist      
    def dist_to_plane(plane_points, test_points, ref_points):
        """
        Calculates the distance between planes and a sets of points.
        
        Calculates the minimum distance between planes and sets of points. 
        The planes are defined by three points each that are part of the respective plane.
        
        Keyword arguments:
            plane_points : np.array [set, plane, point, member/index(4D)]
                Points that define the plane
            test_points : np.array [set, point, member/index(4D)]
                Points whose distance to the line shall be calculated
            ref_points : np.array [set, point, member(3D)]
                Reference points defining which direction is to be considered negative
        Return value:
            Numpy array containing the distance of each point to it's respective plane.
        """
        # Remove indices from points
        plane_points = plane_points[:,:,:,:-1]
        test_points = test_points[:,:,:-1]
        # Dimensions of this calculation
        n_sets = plane_points.shape[0]
        n_planes = plane_points.shape[1]
        n_test_points = test_points.shape[1]
        # Parametric vectors of each plane
        plane_vecs = plane_points[:,:,1:] - np.repeat(plane_points[:,:,0], 2, axis=-2).reshape(n_sets,-1,2,3)
        plane_normals = np.cross(plane_vecs[:,:,0], plane_vecs[:,:,1])
        plane_normals /= la.norm(plane_normals, 2, -1, keepdims=True)
        # Signs with which the final distances will be multiplied 
        signs = np.sign(np.einsum('stm,stm->st', 
                                  (np.repeat(ref_points, n_planes, axis=-2).reshape(n_sets, n_planes, 3) - plane_points[:,:,0]), 
                                  plane_normals))
        diff = np.repeat(test_points, n_planes, axis=0).reshape(n_sets, n_planes, n_test_points, 3) \
            - np.repeat(plane_points[:,:,0], n_test_points, axis=-2).reshape(n_sets, n_planes, n_test_points, 3)
        dists = np.einsum('stpm,spm->stp', diff, plane_normals)
        dists *= np.repeat(signs, n_test_points, axis=-1).reshape(dists.shape)
        return dists
    # Sort points by x,y,z coordinates 
    ind = np.lexsort((points[:,:,2],points[:,:,1],points[:,:,0]))
    for r, row in enumerate(ind):
        points[r] = points[r,row]
    Track which points are allready part of the conves hulls
    part_of_hull = np.full((points.shape[:-1]), False, dtype=bool)
    part_of_hull[np.isnan(points[:,:,0])] = True    # np.nan does not need to become part of the hull
    # Geometric middle of each pointset
    middlepoints = np.nanmean(points[:,:,:-1], axis=-2, keepdims=False)
    # Take extremepoints of each set as first points of convex hull
    first, last = get_first_last_inds(points[:,:,0])
    # Mark as part of hull
    part_of_hull[first[:,0], first[:,1]] = True
    part_of_hull[last[:,0], last[:,1]] = True
    # Find point most distant to the line between the extremepoints 
    first_points = points[first[:,0],first[:,1]]
    last_points = points[last[:,0],last[:,1]]
    line_points = np.concatenate((first_points, last_points), axis=-1).reshape(-1,2,4)
    dists = dist_to_line(line_points, points)
    dist_sort = np.argsort(-dists, axis=-2)
    # Use as next point of convex hull
    new_p_inds = dist_sort[:,0]
    new_p_inds = np.concatenate(
        (np.arange(new_p_inds.shape[0], dtype=np.int64).reshape(-1,1), 
         new_p_inds), axis=-1)
    part_of_hull[new_p_inds[:,0], new_p_inds[:,1]] = True
    new_points = points[new_p_inds[:,0], new_p_inds[:,1]]
    # Create first triangles
    triangles = np.concatenate((first_points, last_points, new_points), axis=-1).reshape(-1,1,3,4)
    # Reiterate until all points are part of the convex hull
    while(~np.all(part_of_hull)):
        dists = dist_to_plane(triangles, points, middlepoints)
        sorted_inds = np.argsort(-dists)
        for s, pointset in enumerate(sorted_inds):
            for t, triangle in enumerate(pointset):
                pass
        break
    # Convert triangles into edges 
    
    # Delete all edges that contain np.nan or two verts of the same square
    
    # Return all edges in index-form