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

# pylint: skip-file

### Top-level SCons script

import os
from os.path import join
from glob import glob

python_versions = ['python3.9']

# Get all folders inside the project
def getSubdirs(path):
	lst = [join(path, name) for name in os.listdir(path) if os.path.isdir(join(path, name)) and name[0] != '.']
	append_list = []
	for subpath in lst:
		append_list.extend(getSubdirs(subpath))
	lst.extend(append_list)
	return lst

folders = sorted(getSubdirs('.'))

# Find all SConscript files
sconscripts = []
for folder in folders:
	sconscripts.extend(Glob(join(folder, 'SConscript')))

for py_version in python_versions:
	# Environment settings
	py_include = os.popen(py_version + ' -m pybind11 --include').read().strip("\n").split(" ")
	py_include = [elem.removeprefix('-I') for elem in py_include]

	env = Environment(
		CC='g++',
		CCFLAGS=['-std=c++17', '-O3', '-shared'],
		CPPPATH=[py_include],
		LIBPATH=[]
	)

	# Execute SConscripts
	SConscript(
		sconscripts,
		exports = ['env', 'py_version']
	)