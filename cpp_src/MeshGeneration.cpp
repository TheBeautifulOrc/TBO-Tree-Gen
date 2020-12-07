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

std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> generate_mesh(const TreeNodeContainer& tnc, const double& base_radius, const double& min_radius, const double& loop_distance, const ushort& interpolation_mode)
{
    // General purpose values
    uint n_nodes = tnc.size();

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

    /*
    Create map, that stores which limbs are connected through which joints
    */
    // Which limbs contain which nodes as starts/ends
    std::map<uint, std::vector<uint>> start_nodes, end_nodes;
    for(uint l = 0; l < limbs.size(); l++)
    {
        auto& curr_limb = limbs.at(l);
        start_nodes[curr_limb.front()].push_back(l);
        end_nodes[curr_limb.back()].push_back(l);
    }
    // Combine the two maps into one comprehensive map
    std::map<uint, std::vector<uint>> joint_map;
    // For each node that terminates a limb
    for(auto& elem : end_nodes)
    {
        const uint& node_ind = elem.first;
        // Check if the node is a starting point for new limbs
        std::vector<uint>& starting_limbs = start_nodes[node_ind];
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
    for(auto& limb : limbs)
    {
        uint n_limb_nodes = limb.size();
        uint n_sections = n_limb_nodes - 1;
        // Container storing addresses of all node locations
        std::vector<const Vector3d*> node_locs(n_limb_nodes);
        for(uint n = 0; n < n_limb_nodes; n++)
        {
            node_locs.at(n) = &tnc.at(limb.at(n)).location;
        }
        // TODO: Create containers for interpolation variables 
        // TODO: Compute both linear and cubic interpolated values
        for(uint s = 0; s < n_sections; s++)
        {
            // Calculate difference vector
            Vector3d diff = *node_locs.at(s+1) - *node_locs.at(s);
            // Distance between nodes
            double dist = diff.norm();
            // Number of points in between nodes necessary to keep distance
            // between edge loops as close as possible to loop_distance
            uint n_points = static_cast<uint>(std::round(dist/loop_distance)) - 1;
            // Offsets between the nodes and neighboring interpolated points
            double offset = (dist - (n_points - 1) * loop_distance) / 2;
        }
        // Cubic interpolation
        if(interpolation_mode == 1 && n_sections > 1)
        {
            Spline3d spline(node_locs);
        }
    }

    return std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>>();
}