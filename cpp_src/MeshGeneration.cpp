// Copyright (C) 2019-2020 Luai "TheBeautifulOrc" Malek

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
#include <Eigen/Dense>
#include "TreeNodes.h"
#include "Utility.h"

// Debug
#include <iostream>
using std::cout;
using std::endl;

using Eigen::Vector3d;

std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> generate_mesh_data(const TreeNodeContainer& tnc, const double& base_radius, const double& min_radius, const double& loop_distance, const ushort& interpolation_mode)
{
    // General purpose values
    const uint n_nodes = tnc.size();

    /*
    "Sweep" square profiles across the "limbs" of the TreeNodeContainer
    */

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
    std::vector<std::vector<uint>> limbs;
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
            limbs.push_back(new_limb);
        }
    }
    const uint n_limbs = limbs.size();

    /*
    Create map, that stores which limbs are connected through which joints
    */
    // Which limbs contain which nodes as starts/ends
    std::map<uint, std::vector<uint>> start_nodes, end_nodes;
    // Table showing if a limb ends on a leaf
    std::vector<bool> leaf_table(n_limbs);
    for(uint l = 0; l < n_limbs; l++)
    {
        const std::vector<uint>& curr_limb = limbs.at(l);
        const uint& front = curr_limb.front();
        const uint& back = curr_limb.back();
        start_nodes[front].push_back(l);
        end_nodes[back].push_back(l);
        leaf_table.at(l) = (tnc.at(back).child_indices.size() == 0);
    }
    // Combine the two maps into one comprehensive map
    std::map<uint, std::vector<uint>> joint_map;
    // For each node that terminates a limb
    for(auto& elem : end_nodes)
    {
        const uint& node_ind = elem.first;
        // Check if the node is a starting point for new limbs
        const std::vector<uint>& starting_limbs = start_nodes[node_ind];
        if(starting_limbs.size() > 0)
        {
            // The first entry of the combined map is always the limb that gets terminated by this node
            std::vector<uint> curr_joint_map = elem.second;
            // Append the limbs that start at this node
            curr_joint_map.insert(curr_joint_map.end(), starting_limbs.begin(), starting_limbs.end());
            // Add generated map to the larger container
            joint_map[node_ind] = curr_joint_map;
        }
    }

    /*
    Calculate points in between connected nodes, necessary to satisfy desired loop distance
    */
    std::vector<std::vector<Vector3d>> centroids(n_limbs);
    for(uint l = 0; l < n_limbs; l++)
    {
        // The current limb
        std::vector<uint>& limb = limbs.at(l);
        // Container for resulting points
        std::vector<Vector3d>& lc = centroids.at(l);
        // Number of nodes in this limb
        const uint n_limb_nodes = limb.size();
        // The number of sections in between nodes 
        const uint n_sections = n_limb_nodes - 1;
        // Container storing addresses of all node locations
        std::vector<const Vector3d*> node_locs(n_limb_nodes);
        for(uint n = 0; n < n_limb_nodes; n++)
        {
            node_locs.at(n) = &tnc.at(limb.at(n)).location;
        }
        // Difference vectors between neighboring nodes
        std::vector<Vector3d> diffs(n_sections);
        // Positions of points in between nodes (in '(0,1)')
        std::vector<std::vector<double>> local_positions(n_sections);
        for(uint s = 0; s < n_sections; s++)
        {
            // Calculate difference vector
            Vector3d& diff = diffs.at(s);
            diff = *node_locs.at(s+1) - *node_locs.at(s);
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
            local_positions.at(s) = section_positions;
        }
        
        /*
        Interpolation
        */
        bool linear_interpolation = (interpolation_mode == 0 || n_sections < 2);
        // Create 3D-spline interpolating this limb...
        Spline3d spline;
        // ... but only perform (costly) initializaiton if necessary
        if(!linear_interpolation)
        {
            spline.init(node_locs);
        }
        // For each section of the limb...
        for(uint s = 0; s < n_sections; s++)
        {
            // References to global data structures for better handling 
            const Vector3d& section_root_node = *node_locs.at(s);
            // Nodes at the front of sections are always centroids
            lc.push_back(section_root_node);
            const std::vector<double>& section_positions = local_positions.at(s);
            const uint n_section_points = section_positions.size();
            std::vector<Vector3d> sp(n_section_points);
            // Linear interpolation
            if(linear_interpolation)
            {
                for(uint p = 0; p < n_section_points; p++)
                {
                    // Each vector is the sum of the location of the node at the beginning
                    // of this section and the vector pointing to the next node weighted by 
                    // the points position relative to the neighboring nodes 
                    sp.at(p) = section_root_node + diffs.at(s) * section_positions.at(p);
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
                    sp.at(p) = spline.evaluate(w);
                }
            }
            // Append results to result variable
            lc.insert(lc.end(), sp.begin(), sp.end());
        }
        // If this limb ends on a leaf node that node is a centroid too
        if(leaf_table.at(l))
        {
            lc.push_back(*node_locs.back());
        }
    }

    // Return data to python interface
    std::vector<Eigen::Vector3d> combined_point_data;
    // Debug
    for(auto& limb_points : centroids)
    {
        combined_point_data.insert(combined_point_data.end(), limb_points.begin(), limb_points.end());
    }
    std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> ret_val;
    std::get<0>(ret_val) = combined_point_data;
    return ret_val;
}