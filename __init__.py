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

bl_info = {
    "name" : "TBO_TreeGenerator",
    "author" : "Luai \"TheBeautifulOrc\" Malek",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 8, 5),
    "location" : "3DView",
    "warning" : "",
    "category" : "Add Mesh" 
}

import bpy
import math
from bpy.props import PointerProperty

if "bpy" in locals():
    # pylint: disable=import-error
    import importlib
    from .py_src import TreeProperties
    importlib.reload(TreeProperties)
    from .py_src import Utility
    importlib.reload(Utility)
    from .py_src import Panels
    importlib.reload(Panels)
    from .py_src import MeshGenration
    importlib.reload(MeshGenration)
    from .py_src import Operators
    importlib.reload(Operators)

classes = (
    TreeProperties.TreeProperties, 
    Operators.CreateTree, 
    Panels.MainPanel, 
    Panels.PRSubPanel,
    Panels.APSubPanel,
    Panels.SCSubPanel,
    Panels.NRSubPanel,
    Panels.SKSubPanel
)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)
    bpy.types.Scene.tbo_treegen_data = PointerProperty(type=TreeProperties.TreeProperties)  # pylint: disable=assignment-from-no-return

def unregister():
    for cl in reversed(classes):
        bpy.utils.unregister_class(cl)
    del bpy.types.Scene.tbo_treegen_data
