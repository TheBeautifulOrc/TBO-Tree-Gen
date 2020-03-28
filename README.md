# TBO-Tree-Gen

This addon for Blender 2.8+ creates trees procedurally implementing a space colonialization algorithm. 

## Introduction

The algorithm implemented in this addon is based on the paper [Modeling Trees with a Space Colonization Algorithm](http://algorithmicbotany.org/papers/colonization.egwnp2007.large.pdf "Link to the paper"). The main advantages of this addon compared to most other tree generation addons for Blender are:

- **Control:** The user knows about the rough shape and size of the tree before anything is being generated. 
- **Tweakability:** The user can change the details of the tree without committing to a new shape or size.
- **Realism:** Since the trees grown by this addon emulate the natural process of competing for space, they will look organic and wont contain artifacts like intersecting branches.  
- **Multiple Trees:** With this workflow multiple tree objects can be grown into one single shape (i.e. a hedge).