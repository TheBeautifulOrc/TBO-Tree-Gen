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

inline bool is_close(const double& a, const double& b) { return (std::abs(a-b) < __DBL_EPSILON__); }
inline bool leq(const double& a, const double& b) { return (a < b) | is_close(a,b); }
inline bool geq(const double& a, const double& b) { return (a > b) | is_close(a,b); }

inline float angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return std::acos(a.dot(b) / (a.norm()*b.norm())); }

Eigen::Vector3d calc_growth_direction(const Eigen::Vector3d& old_node_loc, const std::vector<Eigen::Vector3d>& attr_points);

class Spline;
class Spline3d
{
    // Members
    std::vector<const Eigen::Vector3d*> points;
    std::vector<double> w;
    Spline* y_spline;
    Spline* z_spline;
    Spline* x_spline;

    // Methods
    public:
    Spline3d(std::vector<const Eigen::Vector3d*>& _points);
    double to_w(const double& loc_pos, const uint& seg);
    Eigen::Vector3d evaluate(const double& pos);
    inline std::vector<double> get_w() { return w; };
};