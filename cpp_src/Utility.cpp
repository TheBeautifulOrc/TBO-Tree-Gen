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

#include "Utility.h"
#include <cmath>
#include <tuple>
#include <nanoflann.hpp>

using Eigen::MatrixX3d;
using Eigen::Vector3d;
using nanoflann::KDTreeEigenMatrixAdaptor;
using nanoflann::metric_L2;
using nanoflann::KNNResultSet;
using nanoflann::SearchParams;

bool is_close(double a, double b)
{
    return (std::abs(a-b) < __DBL_EPSILON__);
}

// Nearest neighbor search
const size_t dim = 3;   // Search for 3D data
using kdt = KDTreeEigenMatrixAdaptor<MatrixX3d, dim, metric_L2>;

// Find single nearest neighbor
void find_nearest_neighbor(const Eigen::MatrixX3d& p_matr, const Eigen::Vector3d& query_pt, size_t& res_index, double& res_distance, uint leaf_size)
{
    // Create KDTree
    kdt matr_index(dim, std::ref(p_matr), leaf_size);
    // Query
    KNNResultSet<double> resultSet(1);
    resultSet.init(&res_index, &res_distance);
    matr_index.index->findNeighbors(resultSet, &query_pt[0], SearchParams(10));
    // Take square-root of squared distance
    res_distance = std::sqrt(res_distance);
}

// Find n-nearest neighbors
void find_n_nearest_neighbors(const MatrixX3d& p_matr, const Vector3d& query_pt, std::vector<size_t>& res_indices, std::vector<double>& res_distances, uint leaf_size, uint n_nearest)
{
    // Create KDTree
    kdt matr_index(dim, std::ref(p_matr), leaf_size);
    // Query
    res_indices = std::vector<size_t>(n_nearest);
    res_distances = std::vector<double>(n_nearest);
    KNNResultSet<double> resultSet(n_nearest);
    resultSet.init(&res_indices[0], &res_distances[0]);
    matr_index.index->findNeighbors(resultSet, &query_pt[0], SearchParams(10));
    // Take square-root of squared distances
    for(size_t e = 0; e < res_distances.size(); e++)
    {
        res_distances[e] = sqrt(res_distances[e]);
    }
}

// Calculate normalized vector pointing in the direction of the next growth-step
Vector3d calc_growth_direction(const Vector3d& old_node_loc, const std::vector<Vector3d>& attr_points)
{
    // Number of attraction points influencing growth of this node in this iteration
    size_t n_attr_points = attr_points.size();
    
    // Combine normalized directions by adding
    Vector3d combined_dir(0.0,0.0,0.0);
    for(size_t ap = 0; ap < n_attr_points; ap++)
    {
        combined_dir += (attr_points.at(ap) - old_node_loc).normalized();
    }
    // Combined direction will only be returned if the added vectors aren't too divergent
    // Else branching will be forced
    if(combined_dir.sum() >= 2)
    {
        return combined_dir.normalized();
    }
    else
    {
        return (attr_points.at(0) - old_node_loc).normalized();
    }
}