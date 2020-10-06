# Copyright (C) 2020  Luai "TheBeautifulOrc" Malek

import bpy
import sys
import importlib.util as ilu
import subprocess

# Blender's Python executable
pybin = bpy.app.binary_path_python

def _add_user_site():
    # Locate users site-packages (writable)
    user_site = subprocess.check_output([pybin, "-m", "site", "--user-site"])
    user_site = user_site.decode("utf8").rstrip("\n")   # Convert to string and remove line-break
    # Add user packages to sys.path (if it exits)
    user_site_exists = user_site is not None
    if user_site not in sys.path and user_site_exists:
        sys.path.append(user_site)
    return user_site_exists

def _enable_pip():
    if ilu.find_spec("pip") is None:
        subprocess.check_call([pybin, "-m", "ensurepip", "--user"])
        subprocess.check_call([pybin, "-m", "pip", "install", "--upgrade", "pip", "--user"])
    
def _install_module(module : str):
    if ilu.find_spec(module) is None:
        subprocess.check_call([pybin, "-m", "pip", "install", module, "--user"])
        
def handle_packages(modules):
    user_site_added = _add_user_site()
    _enable_pip()
    for module in modules:
        _install_module(module)
    if not user_site_added:
        _add_user_site()