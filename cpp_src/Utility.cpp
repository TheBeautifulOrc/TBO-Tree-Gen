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

using Eigen::MatrixX3d;
using Eigen::Vector3d;

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