# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
import math
 
import numpy as np
import numpy.linalg as la
import numpy.ma as ma

from dataclasses import dataclass, field
from typing import List
from mathutils import Vector

from .TreeNodes import Tree_Node, Tree_Node_Container
from .Utility import transform_points
from .TreeProperties import TreeProperties
        
class Tree_Object:
    def __init__(self, 
                 bl_object : bpy.types.Object, 
                 nodes : Tree_Node_Container, 
                 tree_data : TreeProperties):
        self.bl_object = bl_object
        self.nodes = nodes
        self.tree_data = tree_data
        
    # Generates the objects skeletal mesh using bmesh
    def generate_skeltal_mesh(self):
        # Calculate vertices in local space 
        verts = [n.location for n in self.nodes]
        tf = self.bl_object.matrix_world.inverted()
        verts = transform_points(tf, verts)
        # Create bmesh
        bm = bmesh.new()
        # Insert vertices
        for v in verts:
            bm.verts.new(v)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        # Create edges
        for p, n in enumerate(self.nodes):
            for c in n.child_indices:
                bm.edges.new((bm.verts[p], bm.verts[c]))
        bm.edges.index_update()
        bm.edges.ensure_lookup_table()
        bm.to_mesh(self.bl_object.data)
        bm.free()
    
    # Generates the objects final mesh using bmesh 
    def generate_mesh(self):
        n_nodes = len(self.nodes)
        # Get node locations
        node_locations = np.array([n.location for n in self.nodes], dtype=np.float64)
        
        # Calculate node radii
        node_weight_factors = np.array([n.weight_factor for n in self.nodes])
        node_radii = np.maximum(np.multiply(node_weight_factors, self.tree_data.sk_base_radius), 
                                self.tree_data.sk_min_radius)
        
        # Calculate list of unit vectors and corresponding distances 
        # pointing from each node to it's connected neigbors
        parents = [0] * n_nodes
        for p, tn in enumerate(self.nodes):
            for c in tn.child_indices:
                parents[c] = p
        neighbor_locs = np.full((n_nodes, 6, 3), np.nan)
        for i, tn in enumerate(self.nodes):
            if i != 0:
                neighbor_locs[i,0,:] = self.nodes[parents[i]].location
            for j, c in enumerate(tn.child_indices):
                k = j+1 if i != 0 else j
                neighbor_locs[i,k,:] = self.nodes[c].location
            
        rel_neighbor_locs = np.subtract(neighbor_locs, 
                                        np.reshape(np.repeat(node_locations, 6, 1), (n_nodes, 6, 3), order='F'))
        neighbor_dists = la.norm(rel_neighbor_locs, 2, -1, keepdims=True)
        neighbor_unit_vecs = np.divide(rel_neighbor_locs, neighbor_dists)
        
        # Calculate local coordinate space for each node
        z_vecs = np.negative(neighbor_unit_vecs[:,0,:]) # Direction to parents is defined as 'down'
        mult_vecs = np.full((n_nodes, 3), [1,0,0])  # Arbitrary vector for multiplication
        mult_vecs[np.allclose(mult_vecs, z_vecs, atol=0)] = [0,1,0] # Ensure that the vectors are different from z_vec
        x_vecs = np.cross(z_vecs, mult_vecs)
        y_vecs = np.cross(z_vecs, x_vecs)
        local_coordinates = np.stack([x_vecs, y_vecs, z_vecs], 1)
        local_coordinates[0] = [[1,0,0], [0,1,0], [0,0,1]]  # Local space of root node always aligns with world space
        
        # Calculate map that assigns a direction relative to local space
        # to each individual neighbor of each node. 
        # The directions are integers mapped like this: 
        #  0,  1,  2,  3,  4,  5
        # +x, -x, +y, -y, +z, -z
        dir_magnitudes = np.einsum('ijl,ikl->jik', local_coordinates, neighbor_unit_vecs)
        abs_magnitudes = np.abs(dir_magnitudes)
        direction_maps = np.full((n_nodes, 6), np.nan)
        for i in range(3):
            mask = np.logical_and(np.greater_equal(abs_magnitudes[i], abs_magnitudes[i-1]), 
                                  np.greater(abs_magnitudes[i], abs_magnitudes[i-2]))
            direction_maps[np.logical_and(mask, np.greater(dir_magnitudes[i], 0))] = i*2
            direction_maps[np.logical_and(mask, np.less(dir_magnitudes[i], 0))] = i*2+1
        
        # Calculate planes: 
        # All planes are defined in normal-form i.e. they have a normal vector defining the orientation of the plane 
        # and a radius (that get's multiplied with the normal to get the base vector). 
        # These planes will be used to calculate the position of the vertices for the final mesh.
        plane_normals = np.full((n_nodes, 6, 3), np.nan, dtype=np.float64)
        plane_radii = np.full((n_nodes, 6), np.nan, dtype=np.float64)
        # Calculate normals:
        # The normals are always the vectors pointing to the adjacent neighbor if present.
        # Else they're the corresponding local coordinate.
        # Create offsets for indexing over the entire map
        map_offset = np.repeat(np.arange(n_nodes), 6).reshape(n_nodes, 6) * 6
        offset_dir_map = direction_maps + map_offset    # Offset the direction map
        existing_dirs = ~np.isnan(offset_dir_map)   # Make a mask for all existing entries in the map
        np.reshape(plane_normals, (-1, 3))[offset_dir_map[existing_dirs].astype(int)] \
            = neighbor_unit_vecs[existing_dirs] # Use neighbor_vectors as normals for planes whenever possible
        missing_entries = np.isnan(plane_normals)   # Create mask for all missing plane normals
        # This array will be multiplied with the local coordinates to recieve an alternating pattern (x+, x-, ...)
        inversion_mask = np.tile(np.array([1,-1,1,-1,1,-1]), (n_nodes, 1)) 
        # Multiply local coordinates with the inversion mask
        directional_coords = np.einsum('ijk,ij->ijk' ,np.repeat(local_coordinates, 2, -2), inversion_mask)
        plane_normals[missing_entries] = directional_coords[missing_entries]    # Fill up the missing entries 
        # Calculate base vectors
        plane_base_radii = np.repeat(node_radii, 6).reshape(n_nodes, 6)
        plane_radii = np.minimum(plane_base_radii, np.divide(neighbor_dists, 4).reshape(n_nodes, 6))
        missing_entries = np.isnan(plane_radii)
        plane_radii[missing_entries] = plane_base_radii[missing_entries]
        
        # Calculate vertices:
        # Each vertex is an intersection point of three of the planes calculated above. Six planes on each node result
        # in a total of 8 intersection points (arranged like the corners of a cube). 
        # The permutation core defines which planes will be intersected with each other for every node.
        # It makes sure that all possible intersections of neighboring planes (never oppsing planes) are calculated.
        # The indices in the core are correspondent to the indices of the direction map.
        permutation_core = np.transpose(np.array([[0,0,0,0,1,1,1,1],[2,3,4,5,2,3,4,5],[5,4,2,3,5,4,2,3]]))
        offset_matrix = np.repeat(np.arange(n_nodes) * 6, 24).reshape(n_nodes, 8, 3)
        permutation_matrix = np.tile(permutation_core, (n_nodes, 1)).reshape(n_nodes, 8, 3)
        permutation_matrix += offset_matrix # Offset the permutation core (make indices for each node unique)
        # Lefthand-side of the linear equations system (x (dot) normal_vec)
        lin_eqs_lhs = np.full((n_nodes, 6, 3, 3), np.nan, dtype=np.float64)
        lin_eqs_lhs = plane_normals.reshape(-1,3)[permutation_matrix]
        # Righthand-side of the linear system (base_vector)
        lin_eqs_rhs = plane_radii.reshape(-1)[permutation_matrix]
        # Solve for x (the resulting intersections)
        rel_verts = la.solve(lin_eqs_lhs, lin_eqs_rhs)
        # Add the result to the initial node locations for non-relative values
        verts = rel_verts + np.repeat(node_locations, 8, axis=0).reshape(n_nodes, 8, 3)
        
        # Create bmesh:
        bm = bmesh.new()
        # Take the calculated vertices and write them into a bmesh (they'll have the same order as before)
        for v in verts.reshape(-1,3):
            bm.verts.new(v)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        
        bm.to_mesh(self.bl_object.data)
        bm.free()