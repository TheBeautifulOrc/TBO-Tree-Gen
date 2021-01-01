# Copyright (C) 2019-2021 Luai "TheBeautifulOrc" Malek

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
import bmesh
from .TreeProperties import TreeProperties
from ..cpp_bin.TreeGenModule import TreeNode, TreeNodeContainer, generate_mesh_data

def genrate_skeletal_mesh(obj : bpy.types.Object, tnc : TreeNodeContainer):
    # Create bmesh-object
    bm = bmesh.new()    # pylint: disable=assignment-from-no-return
    # Generate vertices from TreeNode locations
    [bm.verts.new(tn.location) for tn in tnc]
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    # Generate edges 
    [[bm.edges.new((bm.verts[n], bm.verts[c])) for c in tn.child_indices] for n, tn in enumerate(tnc)]
    bm.edges.index_update()
    bm.edges.ensure_lookup_table()
    # Overwrite objects mesh data with bmesh data
    bm.to_mesh(obj.data)
    # Destroy bmesh
    bm.free()
    
def generate_mesh(obj : bpy.types.Object, tnc : TreeNodeContainer, trd : TreeProperties):
    # Prepare variables
    base_radius = trd.sk_base_radius
    min_radius = trd.sk_min_radius
    loop_dist = trd.sk_loop_distance
    interpolation_mode = 0 if trd.sk_interpolation_mode == "LIN" else 1
    # Call C++ mesh generation function
    p = generate_mesh_data(tnc, base_radius, min_radius, loop_dist, interpolation_mode)[0]
    # Create bmesh-object
    bm = bmesh.new()    # pylint: disable=assignment-from-no-return
    # Generate vertices from TreeNode locations
    [bm.verts.new(elem) for elem in p]
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    # Overwrite objects mesh data with bmesh data
    bm.to_mesh(obj.data)
    # Destroy bmesh
    bm.free()