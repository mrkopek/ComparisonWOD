cmake_minimum_required(VERSION 3.10)
project(pcd2hdf5)

# Find PCL package
find_package(PCL 1.7 REQUIRED)

# Find HDF5 package
find_package(HDF5 REQUIRED COMPONENTS C CXX)


# Include directories for PCL
include_directories(${PCL_INCLUDE_DIRS})

# Include directories for HDF5
include_directories(${HDF5_INCLUDE_DIRS})

# Link directories for PCL and HDF5
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

# Avoid "unused parameter" warnings in PCL
add_definitions(${PCL_DEFINITIONS})

# Define the executable
add_executable(pcd2hdf5 main.cpp)

# Link libraries
target_link_libraries(pcd2hdf5 ${PCL_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
