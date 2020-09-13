# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh

import numpy as np
import math 
import itertools

from mathutils import Vector
from dataclasses import dataclass

from .TreeNodes import Tree_Node, Tree_Node_Container

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
    
    Generates convex hulls in 3D space around all given sets of points. 
    Returns array of all vertices that are connected to form said convex hull. 
    
    Keyword arguments: 
        points : numpy.array 
            Numpy array with structe [set, point, members/index (4D)]
            
    Return value:
        Numpy array with indices of connected vertices in one row.
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
        n_sets, n_points = test_points.shape[:-1]
        n = line_points[:,1] - line_points[:,0]
        n /= la.norm(n, 2, -1, keepdims=True)
        diff = test_points - np.repeat(line_points[:,0], test_points.shape[1], axis=0).reshape(test_points.shape)
        term_1 = np.einsum('spm,sm->sp',diff,n)
        dists = np.abs(diff - np.einsum('sp,sm->spm',term_1,n))
        dists = la.norm(dists, 2, -1, keepdims=True).reshape(n_sets,n_points)
        return dists      
    def dist_to_plane(plane_points, test_points, ref_points=None):
        """
        Calculates the distance between planes and a sets of points.
        
        Calculates the minimum distance between planes and sets of points. 
        The planes are defined by three points each that are part of the respective plane.
        
        Keyword arguments:
            plane_points : np.array [set, plane, point, member/index(4D)]
                Points that define the plane
            test_points : np.array [set, point, member/index(4D)]
                Points whose distance to the line shall be calculated
            ref_points (optional) : np.array [set, member(3D)]
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
        diff = np.repeat(test_points, n_planes, axis=0).reshape(n_sets, n_planes, n_test_points, 3) \
            - np.repeat(plane_points[:,:,0], n_test_points, axis=-2).reshape(n_sets, n_planes, n_test_points, 3)
        dists = np.einsum('stpm,stm->stp', diff, plane_normals)
        # Signs with which the final distances will be multiplied 
        if ref_points is not None:
            signs = -np.sign(np.einsum('stm,stm->st', 
                                    (np.repeat(ref_points, n_planes, axis=-2).reshape(n_sets, n_planes, 3) - plane_points[:,:,0]), 
                                    plane_normals))
            dists *= np.repeat(signs, n_test_points, axis=-1).reshape(dists.shape)
        return dists
    # Remove sets with only one square, as they can not have a "hull"
    n_nan = np.count_nonzero(np.isnan(points[:,:,0]), axis=-1)
    n_valid = points.shape[-2] - n_nan
    points = points[np.greater(n_valid, 4)]
    n_sets, n_points = points.shape[:-1]
    # Start of algorithm by finding two most left and right points 
    # Sort points by x,y,z coordinates 
    lex_ind = np.lexsort((points[:,:,2],points[:,:,1],points[:,:,0]))
    for r, row in enumerate(lex_ind):
        points[r] = points[r,row]
    # Track which points are allready part of the conves hulls
    part_of_hull = np.full((points.shape[:-1]), False, dtype=bool)
    part_of_hull[np.isnan(points[:,:,0])] = True    # np.nan does not need to become part of the hull
    # Take extremepoints of each set as first points of convex hull
    first, last = get_first_last_inds(points[:,:,0])
    first_points = points[first[:,0],first[:,1]]
    last_points = points[last[:,0],last[:,1]]
    # Initial lines
    line_points = np.concatenate((first_points, last_points), axis=-1).reshape(-1,2,4)
    # Mark line points as part of hull
    part_of_hull[first[:,0], first[:,1]] = True
    part_of_hull[last[:,0], last[:,1]] = True
    # Find the point most distant to this line 
    dists = dist_to_line(line_points, points)
    sorted_inds = np.argsort(-dists, axis=-1)
    # Use third point as part of convex hull
    third_point_inds = sorted_inds[:,0].reshape(-1,1)
    third_point_inds = np.concatenate((np.arange(n_sets, dtype=np.int64).reshape(-1,1), third_point_inds), axis=-1)
    part_of_hull[third_point_inds[:,0], third_point_inds[:,1]] = True
    third_points = points[third_point_inds[:,0], third_point_inds[:,1]]
    # Generate first set of triangles 
    # Triangles are stored in shape: [set, triangle, point, members/index (4D)]
    triangles = np.concatenate((line_points.reshape(-1,8), third_points), axis=-1).reshape(-1,1,3,4)
    # Find fourth point using the same method
    dists = np.abs(dist_to_plane(triangles, points))
    sorted_inds = np.argsort(-dists, axis=-1)
    fourth_point_inds = sorted_inds[:,:,0].reshape(-1,1)
    fourth_point_inds = np.concatenate((np.arange(n_sets, dtype=np.int64).reshape(-1,1), fourth_point_inds), axis=-1)
    part_of_hull[fourth_point_inds[:,0], fourth_point_inds[:,1]] = True
    fourth_points = points[fourth_point_inds[:,0], fourth_point_inds[:,1]]
    new_triangles = np.concatenate((np.concatenate((fourth_points, triangles[:,0,0], triangles[:,0,1]), axis=1),
                                    np.concatenate((fourth_points, triangles[:,0,1], triangles[:,0,2]), axis=1),
                                    np.concatenate((fourth_points, triangles[:,0,2], triangles[:,0,0]), axis=1)),
                                   axis=1).reshape(n_sets, 3, 3, 4)
    triangles = np.concatenate((triangles, new_triangles), axis=1)
    # Geometric middle of each initial convex hull
    middlepoints = np.nanmean(np.nanmean(triangles[:,:,:,:-1], axis=-2), axis=-2)
    # Reiterate until all points are part of the convex hull
    i = 0
    while(~np.all(part_of_hull)):
        if i == 1:
            break
        # Create new triangle array
        new_triangles = [[] for _ in range(n_sets)]
        # Calculate distance from each point within a set to each triangle whithin the same set
        dists = dist_to_plane(triangles, points, middlepoints)
        sorted_inds = np.argsort(np.negative(dists))
        # Triangles with only negative, zero-ish or np.nan distances to points can not be part of the algorithm 
        points_below_triangle = np.logical_or(np.isclose(dists, 0), dists < 0)   # Is a distance negative (or close to zero)
        points_isnan = np.isnan(dists)   # Is a distance np.nan
        points_valid = ~np.logical_or(points_below_triangle, points_isnan) # If both are false, the point is valid
        n_triangles = triangles.shape[1]    # Current maximum number of triangles in one set
        for s in range(n_sets): # For each set 
            for t in range(n_triangles):    # For each triangle
                new_point = None    # Variable for next point to be integrated into the convex hull
                n_val_points = np.count_nonzero(points_valid[s,t])  # Count valid point above this triangle
                curr_inds = sorted_inds[s,t]
                for p in range(n_val_points):   # For each valid point p
                    ind = curr_inds[p]
                    if not part_of_hull[s,ind]: # If not already part of convex hull
                        part_of_hull[s,ind] = True  # Mark as part of hull
                        new_point = points[s,ind]   # Store as next point to be integrated
                        break   # Don't check any more points
                old_points = triangles[s,t]         # Get the points of the late triangle
                if new_point is not None:   # If a new candidate for integration was found above this triangle
                    # Replace the old triangle with three new triangles, connecting the old points with the new point      
                    new_triangles[s].extend([np.array(([old_points[0], old_points[1], new_point])),
                                             np.array(([old_points[1], old_points[2], new_point])),
                                             np.array(([old_points[0], old_points[2], new_point]))])
                else:
                    new_triangles[s].append(old_points) # Just keep the old triangle
        new_n_tri = max([len(e) for e in new_triangles])
        triangles = np.full((n_sets, new_n_tri, 3, 4), np.nan)
        for s, pointset in enumerate(new_triangles):
            triangles[s,:len(pointset)] = [e for e in pointset]
        i += 1
    # Convert triangles into edges 
    triangles = triangles[:,:,:,-1] # Only look at indices
    triangles = (triangles[~np.isnan(triangles)].reshape(-1,3))   # Remove nan entries and reshape
    # Reorganize into edges
    edges = np.vstack((np.hstack((triangles[:,0].reshape(-1,1), triangles[:,1].reshape(-1,1))),
                       np.hstack((triangles[:,1].reshape(-1,1), triangles[:,2].reshape(-1,1))),
                       np.hstack((triangles[:,2].reshape(-1,1), triangles[:,0].reshape(-1,1)))))
    # Delete all edges that contain np.nan or two verts of the same square
    div_edges = (edges / 4).astype(np.int64)
    valid_edges = np.not_equal(div_edges[:,0], div_edges[:,1])
    edges = edges[valid_edges].astype(np.int64)
    # Delete double edges 
    edges = np.unique(np.sort(edges, axis=-1), axis=0)
    # Return all edges
    return edges