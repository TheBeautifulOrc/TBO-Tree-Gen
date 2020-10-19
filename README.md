# TBO-Tree-Gen (WIP)

This addon for Blender 2.8+ creates trees procedurally by implementing a space colonialization algorithm. 

---
**Warning: This project is a work in progress (hence it's labeled as WIP). The builds that are being pushed to master are neither functional nor stable. Furthermore there are currently no manuals or wiki entries.** 

**This Addon is not production ready. Do not expect any support when trying to work with it in such an early stage!**

---

## Features

The algorithm implemented in this addon is based on the paper [Modeling Trees with a Space Colonization Algorithm](http://algorithmicbotany.org/papers/colonization.egwnp2007.large.pdf "Link to the paper"). The main advantages of this addon compared to most other tree generation addons for Blender are:

- **Control:** The user is in total control of the shape and size of the tree. Defining the overall shape of the tree and tweaking minor details are clearly separated workflows and will not interfere with one another. 
- **Realism:** Since the trees which are grown by this addon emulate the natural process of competing for space, they will look organic and will not contain artifacts like intersecting branches.  
- **Multiple Trees:** With this workflow multiple trees can be "grown" into one single interconnected shape without clipping into each other (i.e. a hedge consisting of multiple individual plants). 

## Installation

1. Download the repository as a .zip-file.
2. Open Blender 2.8+.
3. Go to `Edit -> Preferences -> Addons` and click the `Install` button in the top-right corner.
4. Select the downloaded .zip-file and you're done.