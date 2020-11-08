# Copyright (C) 2019-2020 Luai "TheBeautifulOrc" Malek

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# pylint: disable=relative-beyond-top-level, 

import bpy
from bpy.types import Object as bpyObject
from mathutils import Vector
from mathutils.kdtree import KDTree

import math
from typing import List

from .TreeNodes import TreeNode, TreeNodeContainer
from .TreeProperties import TreeProperties

def grow_trees(tree_data : TreeProperties, objects : List[bpyObject], p_attr : List[Vector]):
    res_nodes = TreeNodeContainer()
    D = tree_data.sc_D
    d_i = tree_data.sc_d_i * D
    if d_i == 0:
        d_i = math.inf
    d_k = tree_data.sc_d_k * D
    max_iter = tree_data.sc_n_iter if tree_data.sc_n_iter != 0 else math.inf
    # Initialize first nodes 
    [res_nodes.append(TreeNode(obj.location, obj)) for obj in objects]
    # Grow stems
    if d_i != math.inf:
        kdt = KDTree(tree_data.n_p_attr)
        [kdt.insert(p, i) for i, p in enumerate(p_attr)]
        kdt.balance()
        for n in res_nodes:
            n_loc = n.location
            vec, _, dist = kdt.find(n_loc)[:]
            if dist > d_i:
                n_dir = (n_loc - vec).normalized()
                new_loc = vec + n_dir * ((d_i + d_k) / 2)
                n.child_indices.append(len(res_nodes))
                res_nodes.append(TreeNode(new_loc, n.parent_object))
    # Grow crowns 
    continue_growth = True 
    iteration_counter = 0
    while(continue_growth):
        continue_growth = res_nodes.growth_epoch(p_attr, D, d_i, d_k)
        continue_growth = continue_growth and iteration_counter < max_iter
    return res_nodes