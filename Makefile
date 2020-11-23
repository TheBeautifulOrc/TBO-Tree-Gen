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
CC_OPTIONS := -Wall -shared -std=c++11 -fPIC -pipe -fvisibility=hidden
# Filed included during compilation
INCLUDE := $(shell python3.7m -m pybind11 --includes) -I/usr/include/eigen3
# Directories
SRC_DIR := ./cpp_src
OUT_DIR := ./cpp_bin
# Sourcecode
SRC := $(shell find $(SRC_DIR) -name '*.cpp')
# Python extension suffix
PY_SUFF := .cpython-37m-x86_64-linux-gnu.so
# Output Python module
OUT := $(OUT_DIR)/TreeGenModule$(PY_SUFF)

test_build:
	$(CC) $(CC_OPTIONS) $(INCLUDE) -o $(OUT) $(SRC)

clean:
	rm $(OUT)