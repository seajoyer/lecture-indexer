cmake_minimum_required(VERSION 3.10)
project(mlcs_cpp)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Add source files
set(SOURCES
    mlcs_algorithm.cpp
)

# Add header files
set(HEADERS
    mlcs_algorithm.h
)

# Add the Python module
pybind11_add_module(mlcs_cpp mlcs_python_bindings.cpp ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(mlcs_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# For Linux/Mac, we might need to adjust the RPATH settings
if(UNIX AND NOT APPLE)
    set_target_properties(mlcs_cpp PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Installation
install(TARGETS mlcs_cpp
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
