# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import mathutils

from mathutils import Vector
from dataclasses import dataclass, field
from typing import List

@dataclass
class Tree_Node:
    location : Vector
    parent_object : bpy.types.Object
    depth : int = 0
    weight : int = 1
    weight_factor : float = 0.0
    child_indices : List[int] = field(default_factory=list)