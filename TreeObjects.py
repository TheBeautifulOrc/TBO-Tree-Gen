# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
import math
 
import numpy as np
import numpy.linalg as la
import numpy.ma as ma

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
from mathutils import Vector
import itertools

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
    # Non functional!
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
    
    # Generates the objects final mesh using cubes as starting point
    # Non functional!
    def generate_mesh_cubic(self):
        n_nodes = len(self.nodes)
        # Get node locations
        node_locations = np.array([n.location for n in self.nodes], dtype=np.float64)
        
        # Calculate node radii
        node_weight_factors = np.array([n.weight_factor for n in self.nodes])
        node_radii = np.maximum(np.multiply(node_weight_factors, self.tree_data.sk_base_radius), 
                                self.tree_data.sk_min_radius)
        
        # Calculate  list of unit vectors and corresponding distances 
        # pointing from each node to it's connected neigbors
        parents = self.nodes.parent_indices()
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
        np.reshape(plane_radii, (-1))[offset_dir_map[existing_dirs].astype(int)] \
            = np.minimum(plane_base_radii, np.divide(neighbor_dists, 3).reshape(n_nodes, 6))[existing_dirs]
        missing_entries = np.isnan(plane_radii)
        plane_radii[missing_entries] = plane_base_radii[missing_entries]
        
        # Calculate vertices:
        # Each vertex is an intersection point of three of the planes calculated above. 
        # Six planes on each node result in a total of 8 intersection points (arranged like the corners of a cube). 
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
        # Connect vetrices appropriately
        for i in range(n_nodes):
            indices = np.arange(8) + i*8
            curr_verts = [bm.verts[j] for j in indices]
            for j in range(4):
                bm.edges.new((curr_verts[j], curr_verts[j+4]))
            for j in range(2):
                for k in range(2):
                    offset = k+j*4
                    bm.edges.new((curr_verts[2+offset], curr_verts[0+j*4]))
                    bm.edges.new((curr_verts[2+offset], curr_verts[1+j*4]))
            bm.edges.index_update()
            bm.edges.ensure_lookup_table()
        bmesh.ops.holes_fill(bm, edges=bm.edges, sides=6)
                                
        bm.to_mesh(self.bl_object.data)
        bm.free()
    
    # Generates the objects final mesh using metaballs
    # Too slow!
    def generate_mesh_metaballs(self):
        td = self.tree_data
        base_rad = td.sk_base_radius
        min_rad = td.sk_min_radius
        overlap = td.sk_overlap_factor
        
        # Check if min_rad is too small for default resolution
        need_to_scale = min_rad < 1 # Indicates whether the model needs to be upscaled to work with metaballs properly
        
        # Create metaball object
        mball = bpy.data.metaballs.new("TempMBall")
        mball_obj = bpy.data.objects.new("TempMBallObj", mball)
        mball.resolution = .25
        bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)
        
        # Pad the space in between nodes with more metaballs
        for p in self.nodes:
            # Radius of the parent
            p_rad = p.weight_factor * base_rad
            p_rad = p_rad if p_rad > min_rad else min_rad
            # Add initial metaball 
            ele = mball.elements.new()
            ele.co = p.location
            ele.use_negative = False
            ele.radius = p_rad
            # List of child nodes
            cs = [self.nodes[c] for c in p.child_indices]
            for c in cs:    # For each child 
                # Define the line between parent and said child on which the metaballs will be spawned 
                line = (c.location - p.location)
                direction = np.array(line.normalized()) # Direction of the line
                l = line.length # Length of the line
                # Radius of the child
                c_rad = c.weight_factor * base_rad
                c_rad = c_rad if c_rad > min_rad else min_rad
                # Move iteratively along the line and generate new metaballs
                metaball_centers = []
                metaball_radii = []
                curr_center = p_rad / overlap
                exceeded = False # Have we exceeded our target lenght?
                while(not exceeded):
                    if curr_center > l:
                        exceeded = True
                    else:
                        metaball_centers.append(curr_center)
                        curr_rad = p_rad - (p_rad - c_rad) * (curr_center / l)
                        metaball_radii.append(curr_rad)
                        curr_center += curr_rad / overlap
                # Convert lists to numpy arrays for improved performance
                metaball_radii = np.array(metaball_radii)
                metaball_centers = np.array(metaball_centers)
                metaball_centers = np.einsum('i,j->ij', metaball_centers, direction)    # Turn scalars into vectors
                offset = np.tile(np.array(p.location), (metaball_centers.shape[0])).reshape(-1,3)
                metaball_centers += offset  # Move metaballs to the right "global" position
                if need_to_scale:
                    metaball_centers /= min_rad
                    metaball_radii /= min_rad
                #Create metaballs
                for center, rad in zip(metaball_centers, metaball_radii):
                    ele = mball.elements.new()
                    ele.co = center
                    ele.use_negative = False
                    ele.radius = rad
                    
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mball_obj = mball_obj.evaluated_get(depsgraph)
        
        # Store copy of the mesh in a bmesh
        bm = bmesh.new()
        bm.from_mesh(eval_mball_obj.to_mesh())
        
        if need_to_scale:
            bmesh.ops.scale(bm, vec=(min_rad, min_rad, min_rad), space=self.bl_object.matrix_world, verts=bm.verts)
        
        # Delete metaball object
        bpy.data.objects.remove(mball_obj)
        bpy.data.metaballs.remove(mball)
        
        bm.to_mesh(self.bl_object.data)
        bm.free()
        
    # Generates the objects final mesh using metaball-capsules
    # This implementation should be way faster than generate_mesh_metaballs
    # Non functional!
    def generate_mesh_metacapsules(self):
        td = self.tree_data
        base_rad = td.sk_base_radius
        min_rad = td.sk_min_radius
        overlap = td.sk_overlap_factor
        
        # Check if min_rad is too small for default resolution
        need_to_scale = min_rad < 1 # Indicates whether the model needs to be upscaled to work with metaballs properly
        
        # Create metaball object
        mball = bpy.data.metaballs.new("TempMBall")
        mball_obj = bpy.data.objects.new("TempMBallObj", mball)
        mball.resolution = .25
        bpy.context.view_layer.active_layer_collection.collection.objects.link(mball_obj)
        
        # Pad the space in between nodes with more metaballs
        for p in self.nodes:
            # Radius of the parent
            p_rad = p.weight_factor * base_rad
            p_rad = p_rad if p_rad > min_rad else min_rad
            # Add initial metaball 
            ele = mball.elements.new()
            ele.co = p.location if not need_to_scale else p.location / min_rad
            ele.radius = p_rad if not need_to_scale else p_rad / min_rad
            # List of child nodes
            cs = [self.nodes[c] for c in p.child_indices]
            for c in cs:    # For each child 
                # Define the line between parent and said child on which the metaballs will be spawned 
                line = (c.location - p.location)
                l = line.length # Length of the line
                # Radius, position and rotation of the capsule
                rad = c.weight_factor * base_rad
                rad = rad if rad > min_rad else min_rad
                pos = p.location + line/2
                if need_to_scale:
                    rad /= min_rad
                    pos /= min_rad
                    l /= min_rad
                rot = line.rotation_difference((1,0,0))
                # Create meta-capsule
                ele = mball.elements.new()
                ele.type = 'CAPSULE'
                ele.co = pos
                ele.radius = rad
                ele.rotation = rot
                ele.size_x = l
                
        print(len(mball.elements))
                    
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_mball_obj = mball_obj.evaluated_get(depsgraph)
        
        # Store copy of the mesh in a bmesh
        bm = bmesh.new()
        bm.from_mesh(eval_mball_obj.to_mesh())
        
        if need_to_scale:
            bmesh.ops.scale(bm, vec=(min_rad, min_rad, min_rad), space=self.bl_object.matrix_world, verts=bm.verts)
        
        # Delete metaball object
        bpy.data.objects.remove(mball_obj)
        bpy.data.metaballs.remove(mball)
        
        bm.to_mesh(self.bl_object.data)
        bm.free()
    
    # Generates tree-mesh branchwise
    # Non functional!
    def generate_mesh_separate_pieces(self):
        nodes = self.nodes
        
        already_seen = [False for n in nodes]    # Indicates whether a node has been seen by the following routine
        
        branching_points = [nodes[0]]
        branching_points.extend([n for n in nodes if len(n.child_indices) > 1])
        
        branches = []
        
        for p in branching_points:
            for c in p.child_indices:
                if not already_seen[c]:
                    already_seen[c] = True
                    child = nodes[c]
                    curr_branch = [p]
                    while len(child.child_indices) > 0:
                        curr_branch.append(child)
                        if len(child.child_indices) > 1:
                            next_c = sorted(child.child_indices, key=lambda c: nodes[c].weight)[-1]
                            already_seen[next_c] = True
                        else: 
                            next_c = child.child_indices[0]
                        child = nodes[next_c]
                    curr_branch.append(child)
                    branches.append(curr_branch)
        
    # Generates mesh using the reduces
    def generate_mesh_b_mesh(self):
        # Gather general data
        nodes = self.nodes
        tree_data = self.tree_data
        n_nodes = len(nodes)
        
        # Check for empty trees
        if(n_nodes < 2):
            return
        
        # Create a node structure that represents the tree branchwise 
        # Joints are all nodes with more than 2 children
        joints = [0]
        joints.extend([n for n, _ in enumerate(nodes) if len(nodes[n].child_indices) > 1])
        # Limbs are collections of nodes inbetween joints
        limbs = []
        for i in joints:
            p = nodes[i]
            for c in p.child_indices:
                child = nodes[c]
                limb = [p]
                while len(child.child_indices) == 1:
                    limb.append(child)
                    next_c = child.child_indices[0]
                    child = nodes[next_c]
                limb.append(child)
                limbs.append(limb)
        
        # Prepare data as numpy structures for further calculations
        # Initalize numpy arrays
        n_limbs = len(limbs)
        max_nodes_per_limb = max([len(limb) for limb in limbs])
        node_positions = np.full((n_limbs, max_nodes_per_limb, 3), np.nan)
        node_radii = np.full((n_limbs, max_nodes_per_limb), np.nan)
        # Fill the arrays with appropriate data
        for i, l in enumerate(limbs):
            for j, n in enumerate(l):
                node_positions[i,j] = [elem for elem in n.location]
                node_radii[i,j] = n.weight_factor
        node_radii *= tree_data.sk_base_radius
        node_radii[np.less(node_radii, tree_data.sk_min_radius)] = tree_data.sk_min_radius
        # Calculate positions of the bone centers 
        limb_vecs = node_positions[:,1:] - node_positions[:,:-1]
        bone_positions = limb_vecs / 2 + node_positions[:,:-1]
        # List with alternating bone and node positions
        limb_positions = np.full((n_limbs, (2*max_nodes_per_limb) - 1, 3), np.nan)
        limb_positions[:,::2] = node_positions
        limb_positions[:,1::2] = bone_positions
        # List with radii according to limb_positions
        limb_radii = np.full((n_limbs, (2*max_nodes_per_limb) - 1), np.nan)
        limb_radii[:,::2] = node_radii
        bone_radii = (node_radii[:,:-1] + node_radii[:,1:]) / 2
        limb_radii[:,1::2] = bone_radii
        
        # Sweeping preparations
        # Calculate local coordinates, that will determine the orientation of the squares
        local_coordinates = np.full((n_limbs, 3, 3), np.nan)
        global_coordinates = np.array([[0,0,1], [0,1,0], [1,0,0]])
        first_limb_vector = node_positions[:,1] - node_positions[:,0]
        # Local x is the connection between the parent joint and the first element of the limb
        local_coordinates[:,0] = (first_limb_vector / la.norm(first_limb_vector, 2, -1, keepdims=True))
        # Calculate local y and z
        local_coordinates[:,1] = np.cross(np.tile(global_coordinates[2], (n_limbs, 1)), local_coordinates[:,0])
        local_coordinates[:,2] = np.cross(local_coordinates[:,0], local_coordinates[:,1])
        local_coordinates[:,1:] /= la.norm(local_coordinates[:,1:], 2, -1, True)
        close_to_zero = np.array(
            [np.isclose(la.norm(local_coordinates[:,1:], 2, -1, keepdims=True), 0)], 
            dtype=np.bool).reshape(n_limbs, 2)
        local_coordinates[:,1:][close_to_zero] = \
            np.tile(global_coordinates[:-1], (n_limbs, 1)).reshape(n_limbs, 2, 3)[close_to_zero]
            
        # Initialize sweeping
        # Generate first square on each limb
        first_verts_unscaled = np.swapaxes([-np.sign(i - 1.5) * local_coordinates[:,1] + \
            -np.sign((i % 2) - 0.5) * local_coordinates[:,2] for i in range(4)], 0, 1)
        raw_limb_verts = np.full((n_limbs, 2*max_nodes_per_limb-1, 4, 3), np.nan)
        raw_limb_verts[:,0] = first_verts_unscaled
        upper = limb_vecs[:,:-1]    # All but last rows of limb_vecs
        lower = limb_vecs[:,1:]     # All but first rows of limb vecs
        # Calculate rotation-axes  
        limb_rotation_axes = np.full((n_limbs, max_nodes_per_limb, 3), np.nan, dtype=np.float64)
        limb_rotation_axes[:,1:-1] = np.cross(lower, upper)
        limb_rotation_axes /= la.norm(limb_rotation_axes, 2, -1, True)
        limb_rotation_axes[np.isnan(limb_rotation_axes)] = 0
        limb_rotation_axes = np.repeat(limb_rotation_axes, 2, -2)
        limb_rotation_axes = limb_rotation_axes[:,1:,:]
        # Calculate rotation-magnitudes
        limb_rotation_magnitudes = np.full((n_limbs, max_nodes_per_limb), np.nan, dtype=np.float64)
        limb_rotation_magnitudes[:,1:-1] = -np.arccos(np.einsum('ijk,ijk->ij', lower, upper)\
             / (la.norm(upper, 2, -1, True) * la.norm(lower, 2, -1, True)).reshape(n_limbs, -1))
        limb_rotation_magnitudes[np.isnan(limb_rotation_magnitudes)] = 0
        limb_rotation_magnitudes /= 2
        limb_rotation_magnitudes = np.repeat(limb_rotation_magnitudes, 2, -1)
        limb_rotation_magnitudes = limb_rotation_magnitudes[:,1:]
        # Pre-calculate sines and cosines once to save time later on
        sin_limb_rotations = np.sin(limb_rotation_magnitudes)
        cos_limb_rotations = np.cos(limb_rotation_magnitudes)
        # Rotate verts for sweeping
        for l_step in range(max_nodes_per_limb*2 - 2):
            # Setup variables of this iteration
            src_verts = raw_limb_verts[:,l_step]
            rot_axes = limb_rotation_axes[:,l_step]
            # Half steps for the next node and full steps for the next bone get calculated in one go
            cos_rot = cos_limb_rotations[:,l_step]
            sin_rot = sin_limb_rotations[:,l_step]
            # Rotating vectors around axes
            # Einstein indices: 
            # l -> limb
            # v -> vertex
            # m -> member (three members make up a vert)
            term1_1 = np.einsum('l,lm->lm', (1-cos_rot), rot_axes)
            term1_2 = (np.einsum('lm,lvm->lv', rot_axes, src_verts))
            term1 = np.einsum('lm,lv->lvm', term1_1, term1_2)
            term2 = np.einsum('l,lvm->lvm', cos_rot, src_verts)
            term3_1 = np.cross(np.repeat(rot_axes, 4, axis=0).reshape(n_limbs, 4, 3), src_verts)
            term3 = np.einsum('l,lvm->lvm', sin_rot, term3_1)
            rotated_verts = term1 + term2 + term3
            # Write data back into the variable
            raw_limb_verts[:,l_step+1] = rotated_verts
        # Scale rotated verts by multiplying them with their respecive radii 
        local_limb_verts = np.einsum('ij,ijkl->ijkl', limb_radii, raw_limb_verts)
        # Move them to their designated positions to complete the sweeping-process
        limb_verts = local_limb_verts + np.repeat(limb_positions, 4, axis=1).reshape(n_limbs, -1, 4, 3)
        
        # Helper function, that returns the first and last valid square in each branch
        def get_first_and_last():
            non_nan_indices = np.transpose(np.where(~np.isnan(limb_verts[:,:,0,0])))
            splitter = np.flatnonzero(np.diff(non_nan_indices[:,0])) + 1
            separated_indices = np.array_split(non_nan_indices, splitter)
            first_indices = np.array([b[0] for b in separated_indices])
            last_indices = np.array([b[-1] for b in separated_indices])
            return first_indices, last_indices            
        
        # Clean up first square on non-root branches
        limb_verts[1:,0] = np.nan
        # Clean up last square of all branches that have children (non-leaf branches)
        # Find all non-leaf branches
        not_a_leaf = np.array([len(limb[-1].child_indices) != 0 for limb in limbs])
        _, last_inds = get_first_and_last()
        last_inds = last_inds[not_a_leaf]   # Take only the non-leafes 
        limb_verts[last_inds[:,0], last_inds[:,1]] = np.nan
        
        # Clean up all squares that get create artifacts at joints
        limb_first_node = {}    # Maps with the number of the limb as the key and the ID of 
        limb_last_node = {}     # the limbs first/last node as value
        # Fill the maps with data
        for l, limb in enumerate(limbs):
            limb_first_node[l] = id(limb[0])
            limb_last_node[l] = id(limb[-1])
        # Invert the maps
        limb_first_node_inv, limb_last_node_inv = defaultdict(list), defaultdict(list)
        {limb_first_node_inv[v].append(k) for k, v in limb_first_node.items()}
        {limb_last_node_inv[v].append(k) for k, v in limb_last_node.items()}
        # Create lists showing which limbs are part of which joint 
        # (the joint "rests" on the end of the last entry)
        limb_in_joint = [limb_first_node_inv[key] for key in limb_first_node_inv]
        [limb_in_joint[k].extend(limb_last_node_inv[key]) for k, key in enumerate(limb_first_node_inv)]
        [l.reverse() for l in limb_in_joint]
        # Convert to numpy-array for easier handling 
        # (in case of unhomogenous length, pad with np.nan)
        limb_in_joint = \
            np.array(list(itertools.zip_longest(*limb_in_joint, fillvalue=-1)), dtype=np.int).transpose()[1:]
        # Position of all joints
        joint_positions = np.array([nodes[n].location for n in joints])[1:]
        correction_needed = True    # Condition for continuing the while loop
        while(correction_needed):
            correction_needed = False
            first_inds, last_inds = get_first_and_last()    # Get indices involved in joints 
            first_inds = np.append(first_inds, [-1, 0]).reshape(-1,2)   # Append a fallback index (pointing to np.nan)
            # Get array of indices in correct order
            indices = np.full((limb_in_joint.shape[0], limb_in_joint.shape[1], 2), -1, dtype=np.int)
            indices[:,0] = last_inds[limb_in_joint[:,0]]
            indices[:,1:] = first_inds[limb_in_joint[:,1:]] 
            shaped_indices = indices.reshape(-1,2)  # Reshape array for actual use as indices
            vert_sel_shape = (indices.shape[0], 1, indices.shape[1], 4, 3)
            # Selected verts have the shape
            # joint, permutation, square, vertex, member
            sel_verts = limb_verts[shaped_indices[:,0], shaped_indices[:,1]].reshape(vert_sel_shape)
            sel_verts = np.repeat(sel_verts, indices.shape[1], 1)
            # Permutate sel_verts
            for i in range(1, indices.shape[1]):
                sel_verts[:,i] = np.roll(sel_verts[:,i], -i, 1)
            # Local coordinates of all the selected verts (for normal calculation)
            # they have the shape: joint, square, vertex, member
            local_sel_verts = local_limb_verts\
                [shaped_indices[:,0], shaped_indices[:,1]].reshape(indices.shape[0], indices.shape[1], 4, 3)
            # Non-normalized normals of every square
            normals = np.cross(local_sel_verts[:,:,0], local_sel_verts[:,:,1])
            # Points that will be checked against to detect overshadowing.
            # Stored in shape: joint, permutation, vertex/point, member
            compare_points = np.full((indices.shape[0], indices.shape[1], (indices.shape[1]-1)*4+1, 3), np.nan)
            compare_points[:,:,0] = np.repeat(joint_positions, indices.shape[1], axis=0)\
                .reshape(compare_points[:,:,0].shape)
            compare_points[:,:,1:] = sel_verts[:,:,1:].reshape(compare_points.shape[0], compare_points.shape[1], -1, 3)
            # Differences of the compared points
            compare_diffs = compare_points - np.repeat(sel_verts[:,:,0,0], compare_points.shape[-2], axis=-2)\
                .reshape(compare_points.shape)
            # Dot-product between the difference vectors and the normal of each square reveals if all
            # vertices of the other squares are on the "correct" side.
            compare_dots = np.einsum("jpvm,jpm->jpv", compare_diffs, normals)
            compare_signs = np.sign(compare_dots)   # Only the signs are relevant for evaluation
            compare_signs[compare_signs[:,:,0] < 0] *= -1
            compare_signs[np.isnan(compare_signs)] = 1  # Fill nan values, so they don't interfere with the result
            # Returns "True" for squares that don't have to be deleted
            square_valid = np.all(compare_signs > 0, axis=-1)
            delete_indices = indices[~square_valid]
            # Delete all overshadowed squares in the original data-structure
            limb_verts[delete_indices[:,0], delete_indices[:,1]] = np.nan
            correction_needed = ~np.all(square_valid)
        
        # Remove squares that have no geometry information
        # TODO
        
        # Transfer calculated data into the object
        # Remove NaN-values to "filter" invalid/empty vertices out
        printable_limb_verts = limb_verts[~np.isnan(limb_verts)].reshape(-1,3)
        # Create bmesh
        bm = bmesh.new()
        # Generate verts
        for v in printable_limb_verts:
            bm.verts.new(v)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        # Generate edges 
        for i, v in enumerate(printable_limb_verts):
            if i % 4 == 0:
                verts = bm.verts[i:i+4]
                bm.edges.new((verts[0], verts[1]))
                bm.edges.new((verts[1], verts[3]))
                bm.edges.new((verts[3], verts[2]))
                bm.edges.new((verts[2], verts[0]))
        bm.edges.index_update()
        bm.edges.ensure_lookup_table()
        # Overwrite object-mesh
        bm.to_mesh(self.bl_object.data)
        bm.free()