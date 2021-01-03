cmake_minimum_required(VERSION 3.19)
project(TBO_Tree_Gen LANGUAGES CXX)

add_compile_options(-fPIC -O3)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
add_subdirectory(implementation)

pybind11_add_module(TreeGenModule TreeGenModule.cpp)
target_link_libraries(TreeGenModule PRIVATE 
    Eigen3::Eigen 
    TreeNodes 
    MeshGeneration
    )