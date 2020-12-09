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
#include "../cpp_extern/Splines/Spline.h"

using Eigen::MatrixX3d;
using Eigen::Vector3d;

const double root_2 = std::sqrt(2);

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
    if(combined_dir.norm() >= root_2)
    {
        return combined_dir.normalized();
    }
    else
    {
        return (attr_points.at(0) - old_node_loc).normalized();
    }
}

Spline3d::Spline3d(std::vector<const Vector3d*>& _points) : points(_points)
{
    uint n_points = points.size();
    // Function variable of all three splines
    w = std::vector<double>(n_points);
    w.at(0) = 0.0f;
    for(uint p = 1; p < n_points; p++)
    {
        double dist = (*points.at(p) - *points.at(p-1)).norm();
        w.at(p) = dist + w.at(p-1);
    }
    // Deconstruct list of vectors into lists of scalars
    std::vector<double> x(n_points), y(n_points), z(n_points);
    for(uint p = 0; p < n_points; p++)
    {
        const Vector3d* curr_point = points.at(p);
        x.at(p) = curr_point->x();
        y.at(p) = curr_point->y();
        z.at(p) = curr_point->z();
    }
    x_spline->set_points(w, x);
    y_spline->set_points(w, y);
    z_spline->set_points(w, z);
}

double Spline3d::to_w(const double& loc_pos, const uint& seg)
{
    const double& base_val = w.at(seg);
    double offset = (w.at(seg+1) - w.at(seg)) * loc_pos;
    return base_val + offset;
}

Vector3d Spline3d::evaluate(const double& pos)
{
    return Vector3d((*x_spline)(pos), (*y_spline)(pos), (*z_spline)(pos));
}