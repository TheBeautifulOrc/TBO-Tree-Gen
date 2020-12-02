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
#include <vector>
#include <map>
#include <iostream> // Debug
#include <Eigen/Dense>
#include <nanoflann.hpp>

// Debug
using std::cout;
using std::endl;

using Eigen::Vector3d;
using Eigen::MatrixX3d;
using Eigen::Ref;

using nanoflann::KDTreeEigenMatrixAdaptor;
using nanoflann::metric_L2;
using nanoflann::KNNResultSet;
using nanoflann::SearchParams;
using kdt = KDTreeEigenMatrixAdaptor<MatrixX3d, 3, metric_L2>;
using ref_kdt = KDTreeEigenMatrixAdaptor<Ref<MatrixX3d>, 3, metric_L2>;


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

/* 
Take TreeNodeContainer with initial nodes, attraction points and growth variebles
to iteratively grow trees.
*/
void grow_nodes(TreeNodeContainer& nodes, Ref<MatrixX3d> p_attr, double D, uint d_i_fac, uint d_k_fac, uint max_iter)
{
    double d_i = D * d_i_fac;
    double d_k = D * d_k_fac;
    // Influence radius of zero means infinite radius 
    d_i = is_close(d_i, 0.0) ? INFINITY : d_i;
    // Similarly, setting max_iterations to zero means unlimited iterations
    max_iter = (max_iter==0) ? INFINITY : max_iter;

    /* 
    Nearest neighbor search utilities 
    */
    // Nearest neighbor result buffer
    size_t index;
    double distance;
    KNNResultSet<double> resultSet(1);
    // Nearest neighbor search parameters
    SearchParams sp;

    /* 
    If influence radius is not zero, 
    grow stems between initial tree-nodes and attraction-point-cloud where necessary. 
    */
    if(d_i != INFINITY)
    {
        // Generate KDTree
        ref_kdt matr_index(3, std::ref(p_attr), 10);
        matr_index.index->buildIndex();

        // For each node...
        size_t n_nodes = nodes.size();
        for(size_t n = 0; n < n_nodes; n++)
        {
            TreeNode& node = nodes[n];
            // Find nearest attraction point
            resultSet.init(&index, &distance);
            matr_index.index->findNeighbors(resultSet, &node.location[0], sp);
            // Take square-root of squared distance
            distance = std::sqrt(distance);
            // If a stem needs to be grown
            if(distance > d_i)
            {
                // Load closest attraction point into std::vector
                std::vector<Vector3d> attr_loc(1);
                attr_loc.at(0) = p_attr.row(index);
                // Calculate direction of growth
                Vector3d growth_dir = calc_growth_direction(node.location, attr_loc);
                // Generate new TreeNode at appropriate distance
                TreeNode new_node(node.location + growth_dir*(distance - (d_i + d_k) / 2), node.tree_id);
                // Make new entry in parents child_indices and push new node into node-vector
                node.child_indices.push_back(nodes.size());
                nodes.push_back(new_node);
            }
        }
    }

    /* 
    After the stems are taken care of, the crowns can be grown.
    */
    // Mask of all attraction points that get "deleted" during the growth process
    std::vector<bool> p_attr_mask(p_attr.rows(), true);
    // Flag indicates whether the growth is complete and thus should stop
    bool crowns_grown;
    // Current iteration
    size_t curr_iter = 0;
    do
    {
        // Growth will be stopped after this iteration (unless there is reason to continue)
        crowns_grown = true;
        // Number of currently existing nodes
        size_t n_nodes = nodes.size();
        // Which attraction points influence which nodes 
        std::map<size_t, std::vector<size_t>> influence_map;
        // Organize node locations into matrix
        MatrixX3d node_locs(n_nodes, 3);
        for(size_t n = 0; n < n_nodes; n++)
        {
            node_locs.row(n) = nodes.at(n).location;
        }
        // Genrate KDTree
        kdt node_index(3, std::ref(node_locs), 10);
        node_index.index->buildIndex();

        /*
        Find the closest node for each attraction point and add it to the influence map
        if it is close enough. However, if it is too close (closer than the kill distance 'd_k')
        the attraction point will be marked as "invalid" instead an will be ignored in subsequent 
        iterations. 
        */
        // For each (valid) attraction point...
        for(size_t ap = 0; ap < p_attr.rows(); ap++)
        {
            if(p_attr_mask[ap])
            {
                // Find nearest node to attribute point
                resultSet.init(&index, &distance);
                const Vector3d& curr_row = p_attr.row(ap);
                node_index.index->findNeighbors(resultSet, &curr_row[0], sp);
                // KDTree-search retuns squared distance 
                distance = std::sqrt(distance); 
                // If distance is smaller than or equal to kill distance...
                if(distance <= d_k)
                {
                    // ... mark the attraction point as invalid
                    p_attr_mask[ap] = false;
                }
                // If that's not the case, and the distance is smaller than or equal to influence distance...
                else if (distance <= d_i)
                {
                    // ... add the attraction points index to the according node inside the influence map
                    influence_map[index].push_back(ap);
                    // Algorithm needs to reiterate
                    crowns_grown = false;
                }
            }
        }

        /*
        Once the influence map has been updated with all attraction points influencing 
        their respective nodes, the resulting child-nodes can be grown and pushed into the 
        TreeNodeContainer.
        */
        for(auto& influence_pair : influence_map)
        {
            // Parent node
            TreeNode& node = nodes.at(influence_pair.first);
            // Get node location
            Vector3d& node_loc = node.location;
            // Get locations of the influencing attribute points
            std::vector<Vector3d> p_attr_locs;
            for(auto& p_attr_index : influence_pair.second)
            {
                p_attr_locs.push_back(p_attr.row(p_attr_index));
            }
            // Calculate growth direction of the new vector
            Vector3d growth_dir = calc_growth_direction(node_loc, p_attr_locs);
            // Generate new node
            TreeNode new_node((node_loc + growth_dir*D), node.tree_id);
            // Update parent nodes child indices and push new node into container
            node.child_indices.push_back(nodes.size());
            nodes.push_back(new_node);
        }

        /*
        Re-evaluate whether growth should be terminated after this iteration.
        */
        // Terminate if no attraction points influneced the nodes during this iteration
        // (All attraction points are out of reach or within kill distance.)
        crowns_grown |= (influence_map.size() < 1);
        // Terminate if the maximum iteration count has been reached.
        crowns_grown |= (curr_iter >= max_iter);
        curr_iter++;    // Increment current iteration counter
    } while (!crowns_grown);
}

TreeNodeContainer separate_by_id(TreeNodeContainer& nodes, ulong id)
{
    TreeNodeContainer tnc;
    return tnc;
}

void reduce_nodes(TreeNodeContainer& nodes, double reduction_angle)
{

}