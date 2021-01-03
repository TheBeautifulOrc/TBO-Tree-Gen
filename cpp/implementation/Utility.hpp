// Copyright (C) 2019-2021 Luai "TheBeautifulOrc" Malek

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
#include <memory>
#include <Eigen/Core>

inline auto is_close(const double& a, const double& b) -> bool { return (std::abs(a-b) < __DBL_EPSILON__); }
inline auto leq(const double& a, const double& b) -> bool { return (a < b) | is_close(a,b); }
inline auto geq(const double& a, const double& b) -> bool { return (a > b) | is_close(a,b); }

inline auto angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> float { return std::acos(a.dot(b) / (a.norm()*b.norm())); }

auto calc_growth_direction(const Eigen::Vector3d& old_node_loc, const std::vector<Eigen::Vector3d>& attr_points) -> Eigen::Vector3d;

class Spline;
class Spline3d
{
    // Members
    std::vector<const Eigen::Vector3d*> points;
    std::vector<double> w;
    std::unique_ptr<Spline> y_spline;
    std::unique_ptr<Spline> z_spline;
    std::unique_ptr<Spline> x_spline;

    // Methods
    public:
    Spline3d();
    ~Spline3d();
    void init(std::vector<const Eigen::Vector3d*>& _points);
    auto to_w(const double& loc_pos, const uint& seg) -> double;
    auto evaluate(const double& pos) -> Eigen::Vector3d;
    inline auto get_w() -> std::vector<double> { return w; };
};