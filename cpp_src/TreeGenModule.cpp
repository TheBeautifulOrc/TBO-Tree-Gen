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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <Eigen/Core>
#include <string>
#include "TreeNodes.h"
#include "MeshGeneration.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<TreeNode>);

PYBIND11_MODULE(TreeGenModule, m)
{
    py::bind_vector<std::vector<TreeNode>> (m, "TreeNodeVector");

    py::class_<TreeNode> (m, "TreeNode")
        .def(py::init<Eigen::Vector3d, ulong, std::vector<uint>>())
        .def
        (
            "__repr__", [](const TreeNode& t)
            {
                std::string s_loc = "(" + std::to_string(t.location(0)) + "," + std::to_string(t.location(1)) + "," + std::to_string(t.location(2)) + ")";
                std::string s_id = std::to_string(t.tree_id);
                std::string s_weight = std::to_string(t.weight);
                std::string s_w_fac = std::to_string(t.weight_factor);
                std::string s_ch_ind = "[";
                for(uint ind : t.child_indices)
                {
                    s_ch_ind += std::to_string(ind) + ",";
                }
                s_ch_ind += "]";
                return "TreeNode; location: " + s_loc + ", tree_id: " + s_id + ", weight: " + s_weight + ", weight_factor: " + s_w_fac + ", child_indices: " + s_ch_ind;
            }
        )
        .def_readwrite("location", &TreeNode::location)
        .def_readonly("tree_id", &TreeNode::tree_id)
        .def_readonly("weight", &TreeNode::weight)
        .def_readonly("weight_factor", &TreeNode::weight_factor)
        .def_readwrite("child_indices", &TreeNode::child_indices);

    py::class_<TreeNodeContainer> (m, "TreeNodeContainer")
        .def(py::init<std::vector<TreeNode>*>())
        .def_readwrite("nodes", &TreeNodeContainer::nodes)
        .def("calculate_weights", &TreeNodeContainer::calculate_weights)
        .def("grow_nodes", &TreeNodeContainer::grow_nodes)
        .def("separate_by_id", &TreeNodeContainer::separate_by_id)
        .def("reduce_nodes", &TreeNodeContainer::reduce_nodes);
}