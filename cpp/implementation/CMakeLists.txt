find_package(Eigen3 3.3 REQUIRED)
find_package(nanoflann REQUIRED)

# Utility module
add_library(Utility Utility.cpp)
target_include_directories(Utility INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Utility PRIVATE 
    Eigen3::Eigen
)

# TreeNodes module
add_library(TreeNodes TreeNodes.cpp)
target_include_directories(TreeNodes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TreeNodes PRIVATE 
    Eigen3::Eigen
    Utility 
)

# Mesh generation module
add_library(MeshGeneration MeshGeneration.cpp)
target_include_directories(MeshGeneration INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MeshGeneration PRIVATE
    Eigen3::Eigen
    Utility
    TreeNodes
)