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

#pragma once

#include <vector>
#include <Eigen/Core>

bool is_close(double a, double b);

void find_nearest_neighbor(const Eigen::MatrixX3d& p_matr, const Eigen::Vector3d& query_pt, size_t& res_index, double& res_distance, uint leaf_size = 10);

void find_n_nearest_neighbors(const Eigen::MatrixX3d& p_matr, const Eigen::Vector3d& query_pt, std::vector<size_t>& res_indices, std::vector<double>& res_distances, uint n_nearest = 2, uint leaf_size = 10);

Eigen::Vector3d calc_growth_direction(const Eigen::Vector3d& old_node_loc, const std::vector<Eigen::Vector3d>& attr_points);