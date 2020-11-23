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
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::MatrixX3d;
using Eigen::Ref;

TreeNode::TreeNode(Vector3d _location, ulong _tree_id, std::vector<uint> _child_indices) : location(_location), tree_id(_tree_id), child_indices(_child_indices)
{
    weight = 1;
    weight_factor = 0.0;
}

TreeNodeContainer::TreeNodeContainer(std::vector<TreeNode>* _nodes) : nodes(*_nodes) { };

void TreeNodeContainer::calculate_weights()
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

void TreeNodeContainer::grow_nodes(Ref<MatrixX3d>* p_attr, double D, double d_i, double d_k)
{
    
}