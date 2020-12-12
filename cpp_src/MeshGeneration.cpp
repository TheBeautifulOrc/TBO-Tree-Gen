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

#include "MeshGeneration.h"
#include <map>
#include <Eigen/Dense>
#include "TreeNodes.h"
#include "Utility.h"

// Debug
#include <iostream>
using std::cout;
using std::endl;

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Square = std::array<Vector3d, 4>;

const std::array<int, 4> x_permuation {1,-1,-1,1};
const std::array<int, 4> y_permuation {1,1,-1,-1};

class Centroid
{
    public:
    Vector3d location;
    double radius;
    Centroid(Vector3d _location = Vector3d(0.0,0.0,0.0), double _radius = 0.0) : location(_location), radius(_radius) {};
    ~Centroid() {};
};


std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> generate_mesh_data(const TreeNodeContainer& tnc, const double& base_radius, const double& min_radius, const double& loop_distance, const ushort& interpolation_mode)
{
    // General purpose values
    const uint n_nodes = tnc.size();

    /*
    "Sweep" square profiles across the "limbs" of the TreeNodeContainer
    */

    /* 
    Calculate radii of all nodes
    */ 
    std::vector<double> node_radii(n_nodes);
    for (size_t n = 0; n < n_nodes; n++)
    {
        double& curr_radius = node_radii.at(n);
        // Basic formula for nodes radius
        curr_radius = tnc.at(n).weight_factor * base_radius;
        // If the calculated radius would be smaller than the specified minimum
        // replace with the minimum
        curr_radius = (geq(curr_radius, min_radius)) ? curr_radius : min_radius;
    }

    /*
    Separate tree into "limbs"
    Each limbs is the collection of nodes in between two branches.
    */
    // Find branching points (root node is part of this colletion by definition)
    std::vector<uint> branching_point_inds = {0};
    // Check all points for those with more than one child (branching points)
    for(uint n = 1; n < n_nodes; n++)
    {
        if(tnc.at(n).child_indices.size() > 1)
        {
            branching_point_inds.push_back(n);
        }
    }
    // Nested datastructure for storing each limb separately (without copying data)
    std::vector<std::vector<uint>> limb_indices;
    // At each branching point...
    for(auto bpi : branching_point_inds)
    {
        // ... create a new limb for each outgoing branch
        for (auto& c_ind : tnc.at(bpi).child_indices)
        {
            std::vector<uint> new_limb = {bpi};
            uint next_ind = c_ind;
            // Add next nodes while they only have one child (are part of the branch)
            while(tnc.at(next_ind).child_indices.size() == 1)
            {
                new_limb.push_back(next_ind);
                // Next nodes (only) child becomes the new nex node
                next_ind = tnc.at(next_ind).child_indices.at(0);
            }
            // Terminate limb with next branching point or leaf node
            new_limb.push_back(next_ind);
            // Add new limb to the existing list of branches 
            limb_indices.push_back(new_limb);
        }
    }
    const uint n_limbs = limb_indices.size();

    /*
    Create map, that stores which limbs are connected through which joints
    */
    // Which limbs contain which nodes as starts/ends
    std::map<uint, std::vector<uint>> start_nodes, end_nodes;
    // Table showing if a limb ends on a leaf
    std::vector<bool> leaf_table(n_limbs);
    for(uint l = 0; l < n_limbs; l++)
    {
        const std::vector<uint>& curr_limb = limb_indices.at(l);
        const uint& front = curr_limb.front();
        const uint& back = curr_limb.back();
        start_nodes[front].push_back(l);
        end_nodes[back].push_back(l);
        leaf_table.at(l) = (tnc.at(back).child_indices.size() == 0);
    }
    // Combine the two maps into one comprehensive map
    std::map<uint, std::vector<uint>> joint_map;
    // For each node that terminates a limb
    for(auto& elem : end_nodes)
    {
        const uint& node_ind = elem.first;
        // Check if the node is a starting point for new limbs
        const std::vector<uint>& starting_limbs = start_nodes[node_ind];
        if(starting_limbs.size() > 0)
        {
            // The first entry of the combined map is always the limb that gets terminated by this node
            std::vector<uint> curr_joint_map = elem.second;
            // Append the limbs that start at this node
            curr_joint_map.insert(curr_joint_map.end(), starting_limbs.begin(), starting_limbs.end());
            // Add generated map to the larger container
            joint_map[node_ind] = curr_joint_map;
        }
    }

    /*
    Calculate points in between connected nodes, necessary to satisfy desired loop distance
    */
    std::vector<std::vector<Centroid>> centroids(n_limbs);
    for(uint l = 0; l < n_limbs; l++)
    {
        // The current limb indices
        std::vector<uint>& l_limb_indices = limb_indices.at(l);
        // Container for resulting centroids
        std::vector<Centroid>& l_centroids = centroids.at(l);
        // Number of nodes in this limb
        const uint l_n_nodes = l_limb_indices.size();
        // The number of sections in between nodes 
        const uint l_n_sections = l_n_nodes - 1;
        // Container storing addresses of all node locations and radii
        std::vector<const Vector3d*> l_node_locs(l_n_nodes);
        std::vector<const double*> l_node_radii(l_n_nodes);
        for(uint n = 0; n < l_n_nodes; n++)
        {
            l_node_locs.at(n) = &tnc.at(l_limb_indices.at(n)).location;
            l_node_radii.at(n) = &node_radii.at(l_limb_indices.at(n));
        }
        // Difference vectors between neighboring nodes
        std::vector<Vector3d> diffs(l_n_sections);
        // Positions of points in between nodes (in '(0,1)')
        std::vector<std::vector<double>> l_local_positions(l_n_sections);
        for(uint s = 0; s < l_n_sections; s++)
        {
            // Calculate difference vector
            Vector3d& diff = diffs.at(s);
            diff = *l_node_locs.at(s+1) - *l_node_locs.at(s);
            // Distance between nodes
            double dist = diff.norm();
            // Number of points in between nodes necessary to keep distance
            // between edge loops as close as possible to loop_distance
            uint n_points = static_cast<uint>(std::round(dist/loop_distance)) - 1;
            // Offsets between the nodes and neighboring interpolated points
            double offset = (dist - (n_points - 1) * loop_distance) / 2;
            // Calculate local position of intermediate point relative to neighboring nodes
            std::vector<double> section_positions(n_points);
            for(uint p = 0; p < n_points; p++)
            {
                section_positions.at(p) = (offset + (loop_distance * p)) / dist;
            }
            l_local_positions.at(s) = section_positions;
        }
        
        /*
        Interpolation
        */
        bool linear_interpolation = (interpolation_mode == 0 || l_n_sections < 2);
        // Create 3D-spline interpolating this limb...
        Spline3d spline;
        // ... but only perform (costly) initializaiton if necessary
        if(!linear_interpolation)
        {
            spline.init(l_node_locs);
        }
        // For each section of the limb...
        for(uint s = 0; s < l_n_sections; s++)
        {
            // References to global data structures for better handling 
            const Vector3d& section_root_loc = *l_node_locs.at(s);
            const double& section_root_radius = *l_node_radii.at(s);
            const double& section_end_radius = *l_node_radii.at(s+1);
            // Nodes at the front of sections are always centroids
            l_centroids.push_back(Centroid(section_root_loc, section_root_radius));
            const std::vector<double>& section_positions = l_local_positions.at(s);
            const uint n_section_points = section_positions.size();
            std::vector<Centroid> section_centroids(n_section_points);
            // Linear interpolation
            if(linear_interpolation)
            {
                for(uint p = 0; p < n_section_points; p++)
                {
                    // Each vector is the sum of the location of the node at the beginning
                    // of this section and the vector pointing to the next node weighted by 
                    // the points position relative to the neighboring nodes 
                    section_centroids.at(p).location = section_root_loc + diffs.at(s) * section_positions.at(p);
                }
            }
            // Cubic interpolation
            else
            {
                for(uint p = 0; p < n_section_points; p++)
                {
                    // Convert local position and section counter into global position on limb
                    double w = spline.to_w(section_positions.at(p), s);
                    // Evaluate the cubic spline at this point
                    section_centroids.at(p).location = spline.evaluate(w);
                }
            }
            // In any case linearly interpolate the weight of the in between point
            for(uint p = 0; p < n_section_points; p++)
            {
                const double& pos = section_positions.at(p);
                section_centroids.at(p).radius = section_root_radius * (1-pos) + section_end_radius * pos;
            }
            // Append results to result variable
            l_centroids.insert(l_centroids.end(), section_centroids.begin(), section_centroids.end());
        }
        // Last node of each limb is a centroid too
        l_centroids.push_back(Centroid(*l_node_locs.back(), *l_node_radii.back()));
    }

    /*
    Calculate local coordinate systems
    */
    std::vector<std::vector<Matrix3d>> local_coordinate_systems(n_limbs);
    // For each limb...
    for(uint l = 0; l < n_limbs; l++)
    {
        // Centroids of this limb
        std::vector<Centroid>& l_centroids = centroids.at(l);
        uint n_l_centroids = l_centroids.size();    // Number of centroids 
        uint n_sections = n_l_centroids - 1;        // Number of connections in between centroids
        // Local coordinate systems of this limb 
        std::vector<Matrix3d>& l_lcs = local_coordinate_systems.at(l);
        l_lcs = std::vector<Matrix3d>(n_l_centroids);
        // Pointer-container for centroid locations
        std::vector<Vector3d*> l_locations(n_l_centroids);
        for(uint c = 0; c < n_l_centroids; c++)
        {
            Centroid& cent = l_centroids.at(c);
            l_locations.at(c) = &cent.location;
        }
        
        // Tangent of each section 
        std::vector<Vector3d> l_tangents(n_sections);
        for(uint s = 0; s < n_sections; s++)
        {
            l_tangents.at(s) = (*l_locations.at(s+1) - *l_locations.at(s)).normalized();
        }
        // The outer most centroids of each limb use their adjacent sections tangents
        l_lcs.front().row(0) = l_tangents.front();
        l_lcs.back().row(0) = l_tangents.back();
        // The other centroid tangents are the arithmetic mean of the tangents
        // of their adjacent sections
        for(uint t = 0; t < n_sections - 1; t++)
        {
            l_lcs.at(t+1).row(0) = (l_tangents.at(t) + l_tangents.at(t+1)).normalized();
        }

        /*
        With the tangents as the x-axes the complete local 
        coordinate systems can now be calculated
        */
        // Global coordinates
        Matrix3d ref_coords = Matrix3d::Identity();
        for(uint c = 0; c < n_l_centroids; c++)
        {
            Matrix3d& last_coord_sys = (c == 0) ? ref_coords : l_lcs.at(c-1);
            Matrix3d& curr_coord_sys = l_lcs.at(c);
            if(is_close(curr_coord_sys.row(0).dot(last_coord_sys.row(2)), 1))
            {
                curr_coord_sys.row(1) = last_coord_sys.row(0);
                curr_coord_sys.row(2) = last_coord_sys.row(1);
            }
            else
            {
                curr_coord_sys.row(1) = last_coord_sys.row(2).cross(curr_coord_sys.row(0)).normalized();
                curr_coord_sys.row(2) = curr_coord_sys.row(0).cross(curr_coord_sys.row(1)).normalized();
            }
        }


    }

    /*
    Sweep square profile along calculated positions
    */
    std::vector<std::vector<Square>> squares(n_limbs);
    for(uint l = 0; l < n_limbs; l++)
    {
        std::vector<Centroid>& l_centroids = centroids.at(l);
        std::vector<Matrix3d>& l_lcs = local_coordinate_systems.at(l);
        uint n_l_squares = l_lcs.size();
        std::vector<Square>& l_squares = squares.at(l);
        l_squares = std::vector<Square>(n_l_squares);
        for(uint s = 0; s < n_l_squares; s++)
        {
            Centroid& centroid = l_centroids.at(s);
            Vector3d& pos = centroid.location;
            double& rad = centroid.radius;
            Matrix3d& lcs = l_lcs.at(s);
            Square& square = l_squares.at(s);
            for(uint v = 0; v < 4; v++)
            {
                Vector3d dir_vec = (lcs.row(1) * x_permuation.at(v) + lcs.row(2) * y_permuation.at(v)).transpose();
                square.at(v) = pos + rad * dir_vec;
            }
        }
    }

    // Return data to python interface
    std::vector<Eigen::Vector3d> combined_point_data;
    // Debug
    for(auto& l_squares : squares)
    {
        for(auto& square : l_squares)
        {
            combined_point_data.insert(combined_point_data.end(), square.begin(), square.end());
        }
    }
    std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::vector<uint>>> ret_val;
    std::get<0>(ret_val) = combined_point_data;
    return ret_val;
}