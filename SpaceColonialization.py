# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
from bpy.types import Object as bpyObject
from mathutils import Vector

import math
from typing import List

from .TreeNodes import TreeNode, TreeNodeContainer
from .TreeProperties import TreeProperties

def grow_trees(tree_data : TreeProperties, objects : List[bpyObject]):
    res = TreeNodeContainer()
    # Grow stems
    for obj in objects:
        res.append(TreeNode(obj.location, obj))
    res.iterate_growth()