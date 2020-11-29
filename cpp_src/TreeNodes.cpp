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

#include "TreeNodes.h"
#include "Utility.h"
#include <cmath>
#include <iostream> // Debug
#include <Eigen/Dense>
#include <nanoflann.hpp>

// Debug
using std::cout;
using std::endl;

using Eigen::Vector3d;
using Eigen::MatrixX3d;
using Eigen::Ref;

TreeNode::TreeNode(Vector3d _location, ulong _tree_id, std::vector<uint> _child_indices) : location(_location), tree_id(_tree_id), child_indices(_child_indices)
{
    weight = 1;
    weight_factor = 0.0;
}

void calculate_weights(TreeNodeContainer& nodes)
{
    // Ensure that all weights are set back to 1 at first
    for(TreeNode node : nodes)
    {
        node.weight = 1;
    }
    uint n_nodes = nodes.size();
    // Calculate weights of each node
    for(uint n = n_nodes - 1; n >= 0; n--)
    {
        TreeNode& node = nodes[n];
        uint new_weight = 1;
        for(uint child_ind : node.child_indices)
        {
            new_weight += nodes[child_ind].weight;
        }
        node.weight = new_weight;
        node.weight_factor = static_cast<double>(new_weight) / static_cast<double>(n_nodes);
    }
}

void grow_nodes(TreeNodeContainer& nodes, Ref<MatrixX3d> p_attr, double D, uint d_i_fac, uint d_k_fac, uint max_iter)
{
    double d_i = D * d_i_fac;
    double d_k = D * d_k_fac;
    // Influence radius of zero means infinite radius 
    d_i = is_close(d_i, 0.0) ? INFINITY : d_i;
    // Similarly, setting max_iterations to zero means unlimited iterations
    max_iter = (max_iter==0) ? INFINITY : max_iter;
    
    // If influence radius is not zero, 
    // grow stems between initial tree-nodes and pointcloud where necessary
    if(d_i != INFINITY)
    {
        size_t n_nodes = nodes.size();
        for(size_t n = 0; n < n_nodes; n++)
        {
            TreeNode& node = nodes[n];
            size_t index;
            double distance;
            find_nearest_neighbor(p_attr, node.location, index, distance);
            if(distance > d_i)
            {
                std::vector<Vector3d> attr_loc(1);
                attr_loc.at(0) = p_attr.row(index);
                Vector3d growth_dir = calc_growth_direction(node.location, attr_loc);
                TreeNode new_node(node.location + growth_dir*(distance - (d_i + d_k) / 2), node.tree_id);
                node.child_indices.push_back(nodes.size());
                nodes.push_back(new_node);
            }
        }
    }

    // After the stems are taken care of, the crowns can be grown
    
}

TreeNodeContainer separate_by_id(TreeNodeContainer& nodes, ulong id)
{
    TreeNodeContainer tnc;
    return tnc;
}

void reduce_nodes(TreeNodeContainer& nodes, double reduction_angle)
{

}