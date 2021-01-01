// Copyright (C) 2019-2021 Luai "TheBeautifulOrc" Malek

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "MeshGeneration.h"
#include <map>
#include <algorithm>
#include <Eigen/Dense>
#include "TreeNodes.h"
#include "Utility.h"

// Debug
#include <iostream>
using std::cout;
using std::endl;

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Square = std::array<Vector3d, 4>;
using Limb = std::vector<Square>;

const Matrix3d ref_coords = Matrix3d::Identity();
const std::array<int, 4> x_permuation {1,-1,-1,1};
const std::array<int, 4> y_permuation {1,1,-1,-1};

struct Centroid
{
    Vector3d location;
    double radius;
    Centroid(Vector3d _location = Vector3d(0.0,0.0,0.0), double _radius = 0.0) : location(_location), radius(_radius) {};
    ~Centroid() {};
};

struct JointMapEntry
{
    uint limb_index;
    bool begins_with_joint;
    JointMapEntry(uint _limb_index = 0, bool _begins_with_joint = true) : limb_index(_limb_index), begins_with_joint(_begins_with_joint) {};
    ~JointMapEntry() {};
};

struct Joint
{
    bool ending_limb_present = false;
    std::vector<Limb*> limbs;
    // Needs to be a copy, since joints can be defined in between nodes
    Vector3d center_point;
};

auto generate_mesh_data(const TreeNodeContainer& tnc, const double& base_radius, const double& min_radius, const double& loop_distance, const ushort& interpolation_mode) 
    -> std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>>
{
    // General purpose values
    const uint n_nodes = tnc.size();

    /* 
    Calculate radii of all nodes
    */ 
    std::vector<double> node_radii(n_nodes);
    for (size_t n = 0; n < n_nodes; n++)
    {
        double& curr_radius = node_radii.at(n);
        // Basic formula for nodes radius
        curr_radius = tnc.at(n).weight_factor * base_radius;
        // If the calculated radius would be smaller than the specified minimum
        // replace with the minimum
        curr_radius = (geq(curr_radius, min_radius)) ? curr_radius : min_radius;
    }

    /*
    Separate tree into "limbs"
    Each limbs is the collection of nodes in between two branches.
    */
    // Collection of indices, pointing to the nodes of each limb
    std::vector<std::vector<uint>> limb_indices;
    auto separate_into_limbs = [&] ()
    {
        // Find branching points (root node is part of this colletion by definition)
        std::vector<uint> branching_point_inds = {0};
        // Check all points for those with more than one child (branching points)
        for(uint n = 1; n < n_nodes; n++)
        {
            if(tnc.at(n).child_indices.size() > 1)
            {
                branching_point_inds.push_back(n);
            }
        }
        // Nested datastructure for storing each limb separately (without copying data)
        // At each branching point...
        for(auto bpi : branching_point_inds)
        {
            // ... create a new limb for each outgoing branch
            for (auto& c_ind : tnc.at(bpi).child_indices)
            {
                std::vector<uint> new_limb = {bpi};
                uint next_ind = c_ind;
                // Add next nodes while they only have one child (are part of the branch)
                while(tnc.at(next_ind).child_indices.size() == 1)
                {
                    new_limb.push_back(next_ind);
                    // Next nodes (only) child becomes the new nex node
                    next_ind = tnc.at(next_ind).child_indices.at(0);
                }
                // Terminate limb with next branching point or leaf node
                new_limb.push_back(next_ind);
                // Add new limb to the existing list of branches 
                limb_indices.push_back(new_limb);
            }
        }
    };
    separate_into_limbs();
    const uint n_limbs = limb_indices.size();

    /*
    Create map, that stores which limbs are connected through which joints
    */
    // Map showing which limbs are part of which joint
    std::map<uint, std::vector<JointMapEntry>> proto_joint_map; 
    // Map limb indices to the nodes they start/end with
    std::map<uint, std::vector<uint>> start_nodes, end_nodes;
    // Table showing if a limb ends on a leaf   
    std::vector<bool> leaf_table(n_limbs);  
    auto create_limb_maps = [&] ()
    {
        for(uint l = 0; l < n_limbs; l++)
        {
            const std::vector<uint>& curr_limb = limb_indices.at(l);
            const uint& front = curr_limb.front();
            const uint& back = curr_limb.back();
            start_nodes[front].push_back(l);
            end_nodes[back].push_back(l);
            leaf_table.at(l) = (tnc.at(back).child_indices.size() == 0);
        }
        // For each node that terminates a limb
        for(auto& elem : end_nodes)
        {
            const uint& node_ind = elem.first;
            // Check if the node is a starting point for new limbs
            const std::vector<uint>& starting_limbs = start_nodes[node_ind];
            uint n_starting_limbs = starting_limbs.size();
            if(n_starting_limbs > 0)
            {
                // The first entry of the combined map is always the limb that gets terminated by this node
                std::vector<JointMapEntry> curr_joint_map(n_starting_limbs + 1);
                JointMapEntry& first_entry = curr_joint_map.front();
                first_entry.limb_index = elem.second.front();
                first_entry.begins_with_joint = false;
                // Append the limbs that start at this node
                for(uint s_node = 0; s_node < n_starting_limbs; s_node++)
                {
                    curr_joint_map.at(s_node+1).limb_index = starting_limbs.at(s_node);
                }
                // Add generated map to the larger container
                proto_joint_map[node_ind] = curr_joint_map;
            }
        }
    };
    create_limb_maps();

    /*
    "Sweep" square profiles across the "limbs" of the TreeNodeContainer
    */
    // Two dimensional list of squares (four points each) that make up the tree
    std::vector<Limb> squares;
    auto sweep = [&] ()
    {
        /*
        Calculate points in between connected nodes, necessary to satisfy desired loop distance
        */
        std::vector<std::vector<Centroid>> centroids(n_limbs);
        for(uint l = 0; l < n_limbs; l++)
        {
            // The current limb indices
            const std::vector<uint>& l_limb_indices = limb_indices.at(l);
            // Container for resulting centroids
            std::vector<Centroid>& l_centroids = centroids.at(l);
            // Number of nodes in this limb
            const uint l_n_nodes = l_limb_indices.size();
            // The number of sections in between nodes 
            const uint l_n_sections = l_n_nodes - 1;
            // Container storing addresses of all node locations and radii
            std::vector<const Vector3d*> l_node_locs(l_n_nodes);
            std::vector<const double*> l_node_radii(l_n_nodes);
            for(uint n = 0; n < l_n_nodes; n++)
            {
                l_node_locs.at(n) = &tnc.at(l_limb_indices.at(n)).location;
                l_node_radii.at(n) = &node_radii.at(l_limb_indices.at(n));
            }
            // Difference vectors between neighboring nodes
            std::vector<Vector3d> diffs(l_n_sections);
            // Positions of points in between nodes (in '(0,1)')
            std::vector<std::vector<double>> l_local_positions(l_n_sections);
            for(uint s = 0; s < l_n_sections; s++)
            {
                // Calculate difference vector
                Vector3d& diff = diffs.at(s);
                diff = *l_node_locs.at(s+1) - *l_node_locs.at(s);
                // Distance between nodes
                double dist = diff.norm();
                // Number of points in between nodes necessary to keep distance
                // between edge loops as close as possible to loop_distance
                uint n_points = static_cast<uint>(std::round(dist/loop_distance)) - 1;
                // Offsets between the nodes and neighboring interpolated points
                double offset = (dist - (n_points - 1) * loop_distance) / 2;
                // Calculate local position of intermediate point relative to neighboring nodes
                std::vector<double> section_positions(n_points);
                for(uint p = 0; p < n_points; p++)
                {
                    section_positions.at(p) = (offset + (loop_distance * p)) / dist;
                }
                l_local_positions.at(s) = section_positions;
            }
            
            /*
            Interpolation
            */
            bool linear_interpolation = (interpolation_mode == 0 || l_n_sections < 2);
            // Create 3D-spline interpolating this limb...
            Spline3d spline;
            // ... but only perform (costly) initializaiton if necessary
            if(!linear_interpolation)
            {
                spline.init(l_node_locs);
            }
            // For each section of the limb...
            for(uint s = 0; s < l_n_sections; s++)
            {
                // References to global data structures for better handling 
                const Vector3d& section_root_loc = *l_node_locs.at(s);
                const double& section_root_radius = *l_node_radii.at(s);
                const double& section_end_radius = *l_node_radii.at(s+1);
                // Nodes at the front of sections are always centroids
                l_centroids.push_back(Centroid(section_root_loc, section_root_radius));
                const std::vector<double>& section_positions = l_local_positions.at(s);
                const uint n_section_points = section_positions.size();
                std::vector<Centroid> section_centroids(n_section_points);
                // Linear interpolation
                if(linear_interpolation)
                {
                    for(uint p = 0; p < n_section_points; p++)
                    {
                        // Each vector is the sum of the location of the node at the beginning
                        // of this section and the vector pointing to the next node weighted by 
                        // the points position relative to the neighboring nodes 
                        section_centroids.at(p).location = section_root_loc + diffs.at(s) * section_positions.at(p);
                    }
                }
                // Cubic interpolation
                else
                {
                    for(uint p = 0; p < n_section_points; p++)
                    {
                        // Convert local position and section counter into global position on limb
                        double w = spline.to_w(section_positions.at(p), s);
                        // Evaluate the cubic spline at this point
                        section_centroids.at(p).location = spline.evaluate(w);
                    }
                }
                // In any case linearly interpolate the weight of the in between point
                for(uint p = 0; p < n_section_points; p++)
                {
                    const double& pos = section_positions.at(p);
                    section_centroids.at(p).radius = section_root_radius * (1-pos) + section_end_radius * pos;
                }
                // Append results to result variable
                l_centroids.insert(l_centroids.end(), section_centroids.begin(), section_centroids.end());
            }
            // Last node of each limb is a centroid too
            l_centroids.push_back(Centroid(*l_node_locs.back(), *l_node_radii.back()));
        }

        /*
        Calculate local coordinate systems
        */
        std::vector<std::vector<Matrix3d>> local_coordinate_systems(n_limbs);
        for(uint l = 0; l < n_limbs; l++)
        {
            // Centroids of this limb
            std::vector<Centroid>& l_centroids = centroids.at(l);
            uint n_l_centroids = l_centroids.size();    // Number of centroids 
            uint n_sections = n_l_centroids - 1;        // Number of connections in between centroids
            // Local coordinate systems of this limb 
            std::vector<Matrix3d>& l_lcs = local_coordinate_systems.at(l);
            l_lcs = std::vector<Matrix3d>(n_l_centroids);
            // Pointer-container for centroid locations
            std::vector<Vector3d*> l_locations(n_l_centroids);
            for(uint c = 0; c < n_l_centroids; c++)
            {
                Centroid& cent = l_centroids.at(c);
                l_locations.at(c) = &cent.location;
            }
            
            // Tangent of each section 
            std::vector<Vector3d> l_tangents(n_sections);
            for(uint s = 0; s < n_sections; s++)
            {
                l_tangents.at(s) = (*l_locations.at(s+1) - *l_locations.at(s)).normalized();
            }
            // The outer most centroids of each limb use their adjacent sections tangents
            l_lcs.front().row(0) = l_tangents.front();
            l_lcs.back().row(0) = l_tangents.back();
            // The other centroid tangents are the arithmetic mean of the tangents
            // of their adjacent sections
            for(uint t = 0; t < n_sections - 1; t++)
            {
                l_lcs.at(t+1).row(0) = (l_tangents.at(t) + l_tangents.at(t+1)).normalized();
            }

            /*
            With the tangents as the x-axes the complete local 
            coordinate systems can now be calculated
            */
            // Global coordinates
            for(uint c = 0; c < n_l_centroids; c++)
            {
                const Matrix3d& last_coord_sys = (c == 0) ? ref_coords : l_lcs.at(c-1);
                Matrix3d& curr_coord_sys = l_lcs.at(c);
                if(is_close(curr_coord_sys.row(0).dot(last_coord_sys.row(2)), 1))
                {
                    curr_coord_sys.row(1) = last_coord_sys.row(0);
                    curr_coord_sys.row(2) = last_coord_sys.row(1);
                }
                else
                {
                    curr_coord_sys.row(1) = last_coord_sys.row(2).cross(curr_coord_sys.row(0)).normalized();
                    curr_coord_sys.row(2) = curr_coord_sys.row(0).cross(curr_coord_sys.row(1)).normalized();
                }
            }
        }

        /*
        Sweep square profile along calculated positions
        */
        squares = std::vector<Limb>(n_limbs);
        for(uint l = 0; l < n_limbs; l++)
        {
            std::vector<Centroid>& l_centroids = centroids.at(l);
            std::vector<Matrix3d>& l_lcs = local_coordinate_systems.at(l);
            uint n_l_squares = l_lcs.size();
            Limb& l_squares = squares.at(l);
            l_squares = Limb(n_l_squares);
            for(uint s = 0; s < n_l_squares; s++)
            {
                Centroid& centroid = l_centroids.at(s);
                Vector3d& pos = centroid.location;
                double& rad = centroid.radius;
                Matrix3d& lcs = l_lcs.at(s);
                Square& square = l_squares.at(s);
                for(uint v = 0; v < 4; v++)
                {
                    Vector3d dir_vec = (lcs.row(1) * x_permuation.at(v) + lcs.row(2) * y_permuation.at(v)).transpose();
                    square.at(v) = pos + rad * dir_vec;
                }
            }
        }
    };
    sweep();

    /*
    Convert joint map from map of indices to map of pointers to actual geomety 
    */
    std::vector<Joint> joints(proto_joint_map.size());
    uint jmap_counter = 0;
    for(auto& elem : proto_joint_map)
    {
        Joint new_joint;
        uint n_j_limbs = elem.second.size();
        new_joint.limbs = std::vector<Limb*>(n_j_limbs);
        new_joint.center_point = tnc.at(elem.first).location;
        for(uint l = 0; l < n_j_limbs; l++)
        {
            JointMapEntry entry = elem.second.at(l);
            auto limb_address = &squares.at(entry.limb_index);
            new_joint.limbs.at(l) = limb_address;
            new_joint.ending_limb_present |= !entry.begins_with_joint;
        }
        joints.at(jmap_counter) = new_joint;
        jmap_counter++;
    }

    /*
    In order to combine the resulting limbs into one tree the squares at 
    each joint must be checked for "overshadowing". Overshadowing describes 
    the existence of non-convex geometry at the joint.
    */
    auto remove_overshadowed = [&] ()
    {
        bool all_clear = false;
        uint j_counter = 0;
        while(!all_clear)
        {
            // Counter does not get reset, so that only new joints get
            // processed in future iterations
            for(; j_counter < joints.size(); j_counter++)
            {
                // Joint object at this node 
                Joint& j = joints.at(j_counter);
                // All limbs that are part of this joint
                std::vector<Limb*>& j_limbs = j.limbs;
                uint n_j_limbs = j_limbs.size();
                // Offset map pointing showing the number of invalid squares in a limb
                std::vector<uint> offset_map(n_j_limbs, 0);
                // Indicates whether work on this joint is complete
                bool joint_clear = false;
                // Indicates whether all limbs of this joint have squares left to evaluate
                while(!joint_clear)
                {
                    // Check can not be done if there are any degenerate joints
                    bool limbs_valid = true;
                    for(uint l = 0; l < n_j_limbs; l++)
                    {
                        limbs_valid &= offset_map.at(l) < j_limbs.at(l)->size();
                    }
                    if(!limbs_valid)
                    {   
                        break;
                    }
                    // Get relevant squares
                    std::vector<Limb::iterator> squares_to_test(n_j_limbs);
                    for(uint l = 0; l < n_j_limbs; l++)
                    {
                        auto it = j_limbs.at(l)->begin();
                        uint advancement = (l == 0 && j.ending_limb_present) ? (j_limbs.at(l)->size() - (1 + offset_map.at(l))) : offset_map.at(l);
                        std::advance(it, advancement);
                        squares_to_test.at(l) = it;
                    }
                    // Check each square for overshadowing
                    std::vector<bool> overshadowed(n_j_limbs, false);
                    for(uint l = 0; l < n_j_limbs; l++)
                    {
                        // Currently tested square
                        auto curr_square = squares_to_test.at(l);
                        // Normal vector of the current square
                        Vector3d normal = (curr_square->at(1) - curr_square->at(0)).cross(curr_square->at(2) - curr_square->at(0));
                        // Middle of the current square
                        Vector3d midpoint = (curr_square->at(0) + curr_square->at(2)) / 2;
                        // Vector pointing from the center of the square to the center of the joint 
                        Vector3d to_center = j.center_point - midpoint;
                        // If distance to center is close to zero, the square can automatically be discarded
                        if(is_close(to_center.norm(), 0.0))
                        {
                            overshadowed.at(l) = true;
                        }
                        // If not more tests need to be conducted
                        else
                        {
                            // "Direction" of center point relative to square
                            double center_sgn = normal.dot(to_center);
                            // Compare with each other tested square
                            for(uint ll = 0; ll < n_j_limbs; ll++)
                            {
                                // Except for itself...
                                if(ll != l)
                                {
                                    // Other square
                                    auto comp_square = squares_to_test.at(ll);
                                    // For each of the other squares vertices
                                    for(uint v = 0; v < 4; v++)
                                    {
                                        // Selected vertex
                                        Vector3d& comp_vert = comp_square->at(v);
                                        // Vector pointing from center of examined square 
                                        // to the vertex of the other square
                                        Vector3d to_comp_vert = comp_vert - midpoint;
                                        // "Direction" to vertex
                                        double sgn = normal.dot(to_comp_vert);
                                        // If the two directions dont have the same sign, 
                                        // the examined square is by definition overshadowed
                                        overshadowed.at(l) = (overshadowed.at(l) || !geq(center_sgn * sgn, 0.0));
                                    }
                                }
                            }
                        }
                        // If it is overshadowed, the offset needs to be incremented
                        offset_map.at(l) += overshadowed.at(l);
                    }
                    // If any square was overshadowed loop needs to be repeated
                    joint_clear = !std::any_of(overshadowed.begin(), overshadowed.end(), [](bool b){return b;});
                }
                // cout << "\tDetection done" << endl;
                // Remove overshadowed squares from limbs
                for(uint l = 0; l < n_j_limbs; l++)
                {
                    // Limb of which data will be removed
                    auto p_limb = j_limbs.at(l);
                    uint l_size = p_limb->size();
                    // Number of squares to remove from this limb
                    uint& offset = offset_map.at(l);
                    // Boundaries for erasing
                    Limb::iterator l_boundary = p_limb->begin();
                    Limb::iterator r_boundary = p_limb->begin();
                    std::advance(r_boundary, offset);
                    // Move boundaries if this is an ending limb
                    if(l==0 && j.ending_limb_present)
                    {
                        std::advance(l_boundary, l_size - offset);
                        std::advance(r_boundary, l_size - offset);
                    }
                    p_limb->erase(l_boundary, r_boundary);
                }
                // cout << "\tRemoval done" << endl;
            }
            
            // Map of all joints with degenerate (emtpy) limbs that need to
            // be reprocessed or even merged 
            std::map<uint, std::vector<Limb*>> degenerate_map;
            std::map<Limb*, std::vector<uint>> degenerate_connections;
            // Detect degenerate joints
            for(uint j = 0; j < joints.size(); j++)
            {
                uint n_j_limbs = joints.at(j).limbs.size();
                // Check for empty limbs
                for(uint l = 0; l < n_j_limbs; l++)
                {
                    auto limb = joints.at(j).limbs.at(l);
                    if(limb->size() == 0)
                    {
                        // Joints with empty limbs need to be addressed
                        // and reprocessed 
                        degenerate_connections[limb].push_back(j);
                        degenerate_map[j].push_back(limb);
                    }
                }
            }
            
            // Group degenerate joints into networks
            std::vector<std::map<uint, std::vector<Limb*>>> degenerate_networks;
            // Which degenerates have already been tracked inside a network
            std::map<uint, bool> visited;
            for(auto& degenerate_pair : degenerate_map)
            {
                visited[degenerate_pair.first] = false;
            }
            // Iterate and add to network periodically
            for(auto degenerate_pair : degenerate_map)
            {
                bool& joint_visited = visited.at(degenerate_pair.first);
                // If this joint isn't already part of the map
                if(!joint_visited)
                {
                    // Mark it as visited
                    joint_visited = true;
                    // And add a new network to the map
                    degenerate_networks.push_back(std::map<uint, std::vector<Limb*>>());
                    // Add the current degenerate pair to the network
                    auto& curr_network = degenerate_networks.back();
                    curr_network.insert(degenerate_pair);
                    // Indicates whether a new degenerate has been added to the current network
                    bool network_expanded = true;
                    while(network_expanded)
                    {
                        network_expanded = false;
                        // Then check for each joint in the current network...
                        for(auto& pair : curr_network)
                        {
                            // ... for each dead limb on this joint...
                            for(auto l_addr : pair.second)
                            {
                                // ... if it connects to another joint that has not 
                                // been part of a network yet
                                for(auto& ind : degenerate_connections.at(l_addr))
                                {
                                    // If not mark it as visited and add it to the current network
                                    bool& other_joint_visied = visited.at(ind);
                                    if(!other_joint_visied)
                                    {
                                        other_joint_visied = true;
                                        curr_network[ind] = degenerate_map.at(ind);
                                        network_expanded = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
 
            // If there are no degenerate joints with empty limbs
            // the algorithm can be stopped
            all_clear = (degenerate_networks.size() == 0);
            std::vector<uint> kill_list;
            // Take care of degenerate joints
            for(auto& network : degenerate_networks)
            {
                // Find the "base" of this network
                // Container for all dead limbs
                std::vector<Limb*> dead_limbs;
                for(auto& pair : network)
                {
                    dead_limbs.insert(dead_limbs.end(), pair.second.begin(), pair.second.end());
                }
                // Find the index of the joint that is not 
                // positioned at the end of one of these dead limbs
                uint base_ind = 0;
                for(auto& pair : network)
                {
                    // For each degenerate joint index
                    const uint& ind = pair.first;
                    // Check if the note has no ending limbs present
                    // or if the ending limb is not part of the list of 
                    // dead limbs in this network
                    Joint& j = joints.at(ind);
                    if(!j.ending_limb_present || 
                        std::find(dead_limbs.begin(), dead_limbs.end(), j.limbs.at(0)) == dead_limbs.end())
                    {
                        // If so this joint is the "base" of the entire network...
                        base_ind = ind;
                        // ... and the search can be stopped 
                        break;
                    }
                }
                
                // If this network contains more than one degenerated joint
                // merge them into one, in any case remove the dead limbs
                uint n_network = network.size(); // Number of joints in this network
               
                // Make the new joint a copy of the "base" joint of this network
                Joint new_joint = joints.at(base_ind);
                auto& new_limbs = new_joint.limbs;
                for(auto& pair : network)
                {
                    const uint& ind = pair.first; 
                    // Mark old joint to be removed later
                    kill_list.push_back(ind);
                    // For each joint in this network
                    Joint& j = joints.at(ind);
                    // Add its center to amd insert its limbs 
                    // into the new joint
                    // (except in the case of the "base" joint, 
                    // its limbs are already present since the 
                    // new joint is a copy of it)
                    if(ind != base_ind)
                    {
                        new_joint.center_point += j.center_point;
                        new_limbs.insert(new_limbs.end(), j.limbs.begin(), j.limbs.end());
                    }
                    // Remove dead limbs that have been copied over
                    for(auto& dead_limb : pair.second)
                    {
                        new_limbs.erase(std::find(new_limbs.begin(), new_limbs.end(), dead_limb));
                    }
                }
                // Center point of new joint is arithmetic mean of cumulative sum
                new_joint.center_point /= n_network;
                // Append new joint to end of joint list
                joints.push_back(new_joint);
                // Subtract network size from counter variable
                // since that many joints will be removed (and 
                // subsequently shorten the vector)
                j_counter -= n_network;
            }        
            // Sort indices of joints that shall be removed
            std::sort(kill_list.begin(), kill_list.end());
            // Delete unnecessary joints in reverse order
            // to avoid problems with constant indices
            for(int k = kill_list.size() - 1; k >= 0; k--)
            {
                uint& kill_ind = kill_list.at(k);
                auto it = joints.begin();
                std::advance(it, kill_ind);
                joints.erase(it);
            }
        }    
    };
    remove_overshadowed();

    // Return data to python interface
    std::vector<Eigen::Vector3d> combined_point_data;
    // Debug
    for(auto& l_squares : squares)
    {
        for(auto& square : l_squares)
        {
            combined_point_data.insert(combined_point_data.end(), square.begin(), square.end());
        }
    }
    std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> ret_val;
    std::get<0>(ret_val) = combined_point_data;
    return ret_val;
}