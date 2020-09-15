# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

bl_info = {
    "name" : "TBO_TreeGenerator",
    "author" : "Luai \"TheBeautifulOrc\" Malek",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 8, 2),
    "location" : "3DView",
    "warning" : "",
    "category" : "Add Mesh" 
}

import bpy
import math
from bpy.props import PointerProperty

if "bpy" in locals():
    import importlib
    from . import TreeProperties
    importlib.reload(TreeProperties)
    from . import Utility
    importlib.reload(Utility)
    from . import Panels
    importlib.reload(Panels)
    from . import TreeNodes
    importlib.reload(TreeNodes)
    from . import TreeObjects
    importlib.reload(TreeObjects)
    from . import Operators
    importlib.reload(Operators)
    from . import PackageHandler
    importlib.reload(PackageHandler)

classes = (
    TreeProperties.TreeProperties, 
    Operators.CreateTree, 
    Panels.MainPanel, 
    Panels.PRSubPanel,
    Panels.APSubPanel,
    Panels.SCSubPanel,
    Panels.VRSubPanel,
    Panels.SKSubPanel
)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)
    bpy.types.Scene.tbo_treegen_data = PointerProperty(type=TreeProperties.TreeProperties)
    user_site_added = PackageHandler.add_user_site()
    PackageHandler.enable_pip()
    PackageHandler.install_module("numba")
    if not user_site_added:
        PackageHandler.add_user_site()

def unregister():
    for cl in reversed(classes):
        bpy.utils.unregister_class(cl)
    del bpy.types.Scene.tbo_treegen_data
