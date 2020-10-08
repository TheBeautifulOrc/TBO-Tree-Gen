# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import bmesh
 
import numpy as np
import numpy.linalg as la

from dataclasses import dataclass, field
from typing import List
from collections import defaultdict
from mathutils import Vector
import itertools

from .TreeNodes import TreeNode, TreeNodeContainer
from .TreeProperties import TreeProperties
from .Utility import transform_points, quick_hull
        
class TreeObject:
    """
    Class used to combine all relevant data to create the trees mesh.
    
    In order to finalize a tree and create it's mesh in Blender this class 
    combines the data of a TreeNodecontainer and the corresponding Blender 
    object in addition to the TreeProperties variable. 
    """
    def __init__(self, 
                 bl_object : bpy.types.Object, 
                 nodes : TreeNodeContainer, 
                 tree_data : TreeProperties):
        """Constructor.
        
        Keyword arguments:
            bl_object : bpy.types.Object 
                Blender object that will become a tree
            nodes : TreeNodeContainer
                Nodes that make up this tree
            tree_data : TreeProperties 
                List of all properties of the tree that is being worked on
        """
        self.bl_object = bl_object
        self.nodes = nodes
        self.tree_data = tree_data
        
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
        
    def generate_mesh_ji_liu_wang(self):
        """
        Generates mesh from TreeNodes object.
        
        Generates quad-dominant mesh from TreeNodes object utilizing the 
        "BMesh"-algorithm by Ji, Liu and Wang. 
        """
        # Gather general data
        nodes = self.nodes
        tree_data = self.tree_data
        n_nodes = len(nodes)
        
        # Check for empty trees
        if(n_nodes < 2):
            return -1
        
        # Create a node structure that represents the tree branchwise 
        # Joints are all nodes with more than 2 children
        joints = [0]
        joints.extend([n for n, _ in enumerate(nodes) if len(nodes[n].child_indices) > 1])
        # Limbs are collections of nodes in between joints
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
        n_limbs = len(limbs)
        # Check for trees with only one limb
        if(n_limbs < 2):
            return -2
        # Initialize numpy arrays
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
        # Scale rotated verts by multiplying them with their respective radii 
        local_limb_verts = np.einsum('ij,ijkl->ijkl', limb_radii, raw_limb_verts)
        # Move them to their designated positions to complete the sweeping-process
        limb_verts = local_limb_verts + np.repeat(limb_positions, 4, axis=1).reshape(n_limbs, -1, 4, 3)
        
        # Helper functions for cleanup
        def get_limb_verts_first_last():
            """Return the first and last valid square of each branch from the limb_verts."""
            non_nan_indices = np.transpose(np.where(~np.isnan(limb_verts[:,:,0,0])))
            splitter = np.flatnonzero(np.diff(non_nan_indices[:,0])) + 1
            separated_indices = np.array_split(non_nan_indices, splitter)
            first_indices_incomplete = np.array([b[0] for b in separated_indices])
            last_indices_incomplete = np.array([b[-1] for b in separated_indices])
            first_indices = np.full((n_limbs, 2), [-1,0], dtype=np.int)
            last_indices = np.full_like(first_indices, [-1,0])
            for elem_f, elem_l in zip(first_indices_incomplete, last_indices_incomplete):
                first_indices[elem_f[0]] = elem_f
                last_indices[elem_l[0]] = elem_l
            return first_indices, last_indices
        
        def delete_overshadowed():
            """Fill all overshadowed sqaures with np.nan"""
            correction_needed = True    # Condition for continuing the while loop
            while(correction_needed):
                first_inds, last_inds = get_limb_verts_first_last()    # Get indices involved in joints 
                first_inds = np.append(first_inds, [-1, 0]).reshape(-1,2)   # Append a fallback index (pointing to np.nan)
                # Convert limb_in_joint_dict to numpy-array for easier handling 
                # (in case of in-homogenous length, pad with np.nan)
                limb_in_joint = [val for _, val in limb_in_joint_dict.items()]
                limb_in_joint = \
                    np.array(list(itertools.zip_longest(*limb_in_joint, fillvalue=-1)), dtype=np.int).transpose()
                # Get array of indices in correct order
                indices = np.full((limb_in_joint.shape[0], limb_in_joint.shape[1], 2), -1, dtype=np.int)
                indices[:,0] = last_inds[limb_in_joint[:,0]]
                indices[:,1:] = first_inds[limb_in_joint[:,1:]] 
                shaped_indices = indices.reshape(-1,2)  # Reshape array for actual use as indices
                # Selected verts have the shape
                # joint, permutation, square, vertex, member
                vert_sel_shape = (indices.shape[0], 1, indices.shape[1], 4, 3)
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
                relevant_joint_positions = joint_positions[[key for key in limb_in_joint_dict]]
                compare_points[:,:,0] = np.repeat(relevant_joint_positions, indices.shape[1], axis=0)\
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
        
        # Clean up first square on non-root branches
        limb_verts[1:,0] = np.nan
        # Clean up last square of all branches that have children (non-leaf branches)
        # Find all non-leaf branches
        not_a_leaf = np.array([len(limb[-1].child_indices) != 0 for limb in limbs])
        _, last_inds = get_limb_verts_first_last()
        last_inds = last_inds[not_a_leaf]   # Take only the non-leafes 
        limb_verts[last_inds[:,0], last_inds[:,1]] = np.nan
        
        # Clean up all squares that get create artifacts at joints
        # Cleanup preparations...
        # Which limbs are connected via joints?
        limb_first_node = {}    # Maps with the number of the limb as the key  
        limb_last_node = {}     # and the ID of the limbs first/last node as value
        # Fill the maps with data
        for l, limb in enumerate(limbs):
            limb_first_node[l] = id(limb[0])
            limb_last_node[l] = id(limb[-1])
        # Invert the maps
        limb_first_node_inv, limb_last_node_inv, limb_in_joint_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        {limb_first_node_inv[v].append(k) for k, v in limb_first_node.items()}
        {limb_last_node_inv[v].append(k) for k, v in limb_last_node.items()}
        # Combine the inverted dicts to create a dict containing all limbs connected by one particular joint
        for k, key in enumerate(limb_first_node_inv):
            limb_in_joint_dict[k] = sorted(limb_first_node_inv[key] + limb_last_node_inv[key])
        joint_positions = np.array([nodes[n].location for n in joints])
        joints_collapsed = True # Track whether routine needs to be rerun
        while(joints_collapsed):
            # Delete all squares that are overshadowed and thus unfit to be part of a convex hull
            # This step needs to be re-run every time joints are merged 
            delete_overshadowed()
            # Combine all joints that are connected with branches that are empty
            # Find all limb-indices that have been completely deleted 
            dead_limb_inds = np.where(np.all(np.isnan(limb_verts).reshape(n_limbs, -1), axis=-1))[0].tolist()
            inv_limb_joint_dict = defaultdict(list)
            {inv_limb_joint_dict[v].append(k) for k in limb_in_joint_dict for v in limb_in_joint_dict[k]}
            # Find joints that need to be combined
            joints_with_dead_limbs = {key : inv_limb_joint_dict[key] for key in dead_limb_inds}
            # Only two joints connected by an empty limb can be joined  
            joints_to_combine = {key : val for key, val in joints_with_dead_limbs.items() if len(val) == 2}
            joints_collapsed = len(joints_to_combine) > 0
            # Combine selected joints
            for limb, joints in joints_to_combine.items():
                first, second = joints
                new_entries = [e for e in limb_in_joint_dict[first] + limb_in_joint_dict[second] if e != limb]
                limb_in_joint_dict[first] = new_entries
                limb_in_joint_dict.pop(second)
            # Delete all other references to dead limbs
            joints_to_clean = {key : val for key, val in joints_with_dead_limbs.items() if len(val) == 1}
            for limb, joints in joints_to_clean.items():
                limb_in_joint_dict[joints[0]] = [e for e in limb_in_joint_dict[joints[0]] if e != limb]
        # Sort lists in dicts, so it can be assumed that the first entry is the end of the base limb
        [limb_in_joint_dict[key].sort() for key in limb_in_joint_dict]
            
        # Remove squares that have no geometry information
        # Synchronize verts in local coords with the previous deletions
        local_limb_verts[np.isnan(limb_verts)] = np.nan
        # Calculate normals
        normals = np.cross(local_limb_verts[:,:,0], local_limb_verts[:,:,1])
        normals /= la.norm(normals, 2, -1, True)
        # Calculate differences of the normals to the normales of the previous and next squares
        diff_to_prev = np.diff(normals, axis=-2)[:,:-1]
        diff_to_next = np.flip(np.diff(np.flip(normals, axis=-2), axis=-2)[:,:-1], axis=-2)
        # If these differences are close to zero, the squares are (roughly) parallel
        parallel_to_prev = np.all(np.isclose(diff_to_prev, 0, atol=1.e-2), axis=-1)
        parallel_to_next = np.all(np.isclose(diff_to_next, 0, atol=1.e-2), axis=-1)
        # If a square is parallel to the previous and next square, it is considered redundant
        square_redundant = np.logical_and(parallel_to_prev, parallel_to_next)
        # Delete redundant squares
        limb_verts[:,1:-1][square_redundant] = np.nan
            
        # Transfer calculated data into the object
        # Remove NaN-values to "filter" invalid/empty vertices out
        printable_limb_verts = limb_verts[~np.isnan(limb_verts)].reshape(-1,3)
        # Index list for these unordered verts
        raw_index_list = np.cumsum(np.array([np.count_nonzero(~np.isnan(limb[:,0,0])) for limb in limb_verts]))
        raw_index_list = np.concatenate((np.array([0]), raw_index_list))
        index_list = np.unique(raw_index_list)
        n_squares = index_list[-1]
        # Create bmesh
        bm = bmesh.new()
        # Generate verts
        for v in printable_limb_verts:
            bm.verts.new(v)
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        # Generate edges
        # Edges within squares 
        for square in range(n_squares):
            vert = square * 4
            verts = bm.verts[vert:vert+4]
            bm.edges.new((verts[0], verts[1]))
            bm.edges.new((verts[1], verts[3]))
            bm.edges.new((verts[3], verts[2]))
            bm.edges.new((verts[2], verts[0]))
        # Edges connecting different squares
        index_counter = 0
        for square in range(n_squares):
            if square != index_list[index_counter]:
                vert = square * 4
                last_verts = bm.verts[vert-4:vert]
                curr_verts = bm.verts[vert:vert+4]
                for lv, cv in zip(last_verts, curr_verts):
                    bm.edges.new((lv, cv))
            else: 
                index_counter += 1
        bm.edges.index_update()
        bm.edges.ensure_lookup_table()
        # Fill geometry with faces 
        bmesh.ops.contextual_create(bm, 
                                    geom=[edge for edge in bm.edges], 
                                    mat_nr=0,
                                    use_smooth=False)
        # Indices of verts and edges that are part of the joints 
        hull_inds = []
        for key, val in limb_in_joint_dict.items():
            hull_inds.append([raw_index_list[v]*4+i for v in val[1:] for i in range(4)])
            #TODO: find out why val is empty in some seeds
            first_entry = (raw_index_list[val[0]+1] - 1) * 4 if key > 0 else (raw_index_list[val[0]]) * 4
            hull_inds[-1].extend([first_entry+i for i in range(4)])
        # Store as numpy array for easier handling
        hull_inds = \
            np.array(list(itertools.zip_longest(*hull_inds, fillvalue=-1)), dtype=np.int).transpose()
        # Group all vertices that make up a square
        hull_inds = hull_inds.reshape((hull_inds.shape[0],-1,4))
        # Identify all faces that need to be deleted
        face_id_dict = {id(face) : face for face in bm.faces}   # ID -> Face
        faces_to_delete = [] 
        # Get list of IDs of all faces that need to be deleted 
        for square in hull_inds[hull_inds > -1].reshape((-1,4)):
            edges = [bm.edges[i] for i in square[:2]]
            linked_faces = [edge.link_faces for edge in edges]
            linked_face_ids = [[id(face) for face in seq] for seq in linked_faces]
            del_face = list(set(linked_face_ids[0]).intersection(linked_face_ids[1]))[0]
            faces_to_delete.append(del_face)
        faces_to_delete = list(set(faces_to_delete))    # Remove duplicates 
        faces_to_delete = [face_id_dict[e] for e in faces_to_delete]    # Turn IDs back into faces
        # Delete unecessary faces
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES_ONLY")
        # Create convex hulls around the joints
        hull_verts = np.array([[c for c in bm.verts[e].co] if e != -1 else [np.nan, np.nan, np.nan] 
                      for e in hull_inds.reshape(-1)]).reshape((hull_inds.shape[0], -1, 3))
        hull_inds = hull_inds.astype(np.float64)
        hull_inds[np.isclose(hull_inds, -1.0)] = np.nan
        hull_verts = np.concatenate((hull_verts, hull_inds.reshape((hull_inds.shape[0], -1, 1))), axis=-1)
        hull_edge_inds = quick_hull(hull_verts)
        for edge_inds in hull_edge_inds:
            bm.edges.new((bm.verts[edge_inds[0]], bm.verts[edge_inds[1]]))
        # Fill holes 
        # bmesh.ops.contextual_create(bm, 
                                    # geom=[edge for edge in bm.edges], 
                                    # mat_nr=0,
                                    # use_smooth=False)
        # Join triangles 
        # bmesh.ops.join_triangles(bm, 
        #                          faces=[face for face in bm.faces], 
        #                          cmp_seam=False,
        #                          cmp_sharp=False,
        #                          cmp_uvs=False,
        #                          cmp_vcols=False,
        #                          cmp_materials=False,
        #                          angle_face_threshold=45.0,
        #                          angle_shape_threshold=45.0)
        # Recalculate normals
        # bmesh.ops.recalc_face_normals(bm, faces=[face for face in bm.faces])
        # Overwrite object-mesh
        bm.to_mesh(self.bl_object.data)
        bm.free()