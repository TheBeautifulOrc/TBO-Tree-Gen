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

### Top-level SCons script

# pylint: skip-file

import os
from os.path import join
from glob import glob

# Suppress SCons warnings
SetOption('warn', 'no-all')

# Requested Python version
AddOption(
	'--py_version', '--py',
	dest='py_version',
	type='string',
	nargs=1,
	action='store',
	default='Python3.9',
	help='''Python version for which C++ code shall be built.\n
		Defaults to Python3.9.\n
		Type "all" to build for all available Python versions.)'''
)
py_selection = GetOption('py_version')
py_selection = py_selection.lower()
to_replace = [' ', '.', '_', '/', 'm', '\n']
for c in to_replace:
	py_selection = py_selection.replace(c, '')

# Detect available Python versions
possible_py_versions = ['python3.7', 'python3.8', 'python3.9']
available_py_versions = []
for v in possible_py_versions:
	version_echo = os.popen(v + ' --version').read()
	version_echo = version_echo.strip('\n')
	version_echo = version_echo.replace(' ', '')
	version_echo = version_echo.lower()
	if version_echo[:len(v)] == v:
		available_py_versions.append(v)

# Match requested Python version against available versions
python_versions = []
if py_selection == 'all':
	python_versions = available_py_versions
else:
	for available in available_py_versions:
		if available.replace('.','') == py_selection:
			python_versions.append(available)

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

vendor_include = [Dir(f.path) for f in os.scandir('cpp/vendor')]
src_include = [Dir('#/cpp/src')]

# Environment settings
env = Environment(
	CC='g++',
	CCFLAGS=['-std=c++17', '-O3'],
	CPPPATH=vendor_include + src_include,
	LIBPATH=[],
	SHLIBPREFIX=''
)

# For each selected Python version...
for py_version in python_versions:
	# ...execute SConscript
	SConscript(
		sconscripts,
		exports=['env', 'py_version'],
		variant_dir=join('cpp', 'build'),
		duplicate=False
	)