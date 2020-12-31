# Copyright (C) 2019-2020 Luai "TheBeautifulOrc" Malek

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

# Compiler
CC := g++
CC_options := -Wall -shared -std=c++11 -O3 -fPIC -pipe -fvisibility=hidden
# Filed included during compilation
3.7_include := $(shell python3.7m -m pybind11 --includes) -I/usr/include/eigen3
3.8_include := $(shell python3.8 -m pybind11 --includes) -I/usr/include/eigen3
# Directories
src_dir := ./cpp_src
out_dir := ./cpp_bin
# Sourcecode
src := $(shell find $(src_dir) -name '*.cpp')
# Python extension suffix
3.7_py_suff := .cpython-37m-x86_64-linux-gnu.so
3.8_py_suff := .cpython-38-x86_64-linux-gnu.so
# Output Python module
3.7_out := $(out_dir)/TreeGenModule$(3.7_py_suff)
3.8_out := $(out_dir)/TreeGenModule$(3.8_py_suff)

test_build:
	$(CC) $(CC_options) $(3.8_include) -o $(3.8_out) $(src)

build_all:
	$(CC) $(CC_options) $(3.7_include) -o $(3.7_out) $(src)
	$(CC) $(CC_options) $(3.8_include) -o $(3.8_out) $(src)

clean:
	rm $(out_dir)/*.so