# Mesh generation algorithm

## Arguments:

- arg0 = joint_map
- arg1 = node_locs 
- arg2 = node_weights
- arg3 = loop_distance
- arg4 = interpolation_mode (default='LINEAR')

## Implementation:

1. **b_mesh_ji_liu_wang**(arg0, arg1, arg2, arg3, arg4) -> verts, connection_table
        
    *foreach limb:*
        
    1. **calc_in_between_locs**(arg1, arg3, arg4) -> locs, dists_to_nodes

    2. **calc_in_between_weights**(arg2, dists_to_nodes) -> weights

    3. **calc_tangents**(locs) -> tangents

    4. **calc_loc_coordinates**(tangents) -> loc_coordinates

    5. **calc_verts**(loc_coordinates, weights) -> verts
        
    2. **stitch**(arg0, verts) -> verts, connection_table

        *foreach joint:*

        1. **eliminate_overshadowed**(arg0, verts) -> verts

        2. **calculate_joint_connections**(arg0, verts) -> connection_table

        *foreach limb:*

        3. **calculate_limb_connections**(arg0, verts, connection_table) -> connection_table