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

#include <Eigen/Core>
#include <pybind11/eigen.h>

class TreeNode
{
    // Members
    public:
    Eigen::Vector3d location;
    ulong tree_id;
    std::vector<uint> child_indices;
    uint weight;
    double weight_factor;

    // Functions
    public:
    TreeNode(Eigen::Vector3d _location, ulong _tree_id, std::vector<uint> _child_indices);
};

class TreeNodeContainer
{
    public:
    // Members
    std::vector<TreeNode> nodes;

    //Functions
    TreeNodeContainer(std::vector<TreeNode>* _nodes);
    void calculate_weights();
    void grow_nodes(Eigen::Ref<Eigen::MatrixX3d>* p_attr, double D, double d_i, double d_k);
    TreeNodeContainer separate_by_id(ulong id);
    void reduce_nodes(double reduction_angle);
};